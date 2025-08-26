import asyncio
import aiohttp
import aiofiles
from aiohttp import web
from PIL import Image
import json
import os.path
import time
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default configuration
PORT = 8000
MAX_CONCURRENT_DOWNLOADS = 10
UPDATE_INTERVAL = 50  # Update every 10 seconds
NUMBER_OF_ACCOUNTS=3
# Helper function to add CORS headers
def add_cors_headers(response: web.Response) -> web.Response:
    """Add CORS headers to a response"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response
def load_config():
    """Load configuration from config.json or use defaults"""
    try:
        if os.path.exists('server_config.json'):
            with open('server_config.json', 'r') as f:
                config = json.load(f)
                return {
                    'port': config.get('port', PORT),
                    'max_concurrent_downloads': config.get('max_concurrent_downloads', MAX_CONCURRENT_DOWNLOADS),
                    'update_interval': config.get('update_interval', UPDATE_INTERVAL)
                }
    except Exception as e:
        logger.warning(f"Could not load server_config.json, using defaults: {e}")
    
    return {
        'port': PORT,
        'max_concurrent_downloads': MAX_CONCURRENT_DOWNLOADS,
        'update_interval': UPDATE_INTERVAL
    }

class TileCache:
    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.last_update = 0
        
    def is_valid(self, tile_key: str) -> bool:
        if tile_key not in self.cache:
            return False
        return time.time() - self.cache[tile_key]['timestamp'] < 300  # 5 minutes cache
    
    def get(self, tile_key: str) -> Dict:
        return self.cache.get(tile_key, {})
    
    def set(self, tile_key: str, data: Dict):
        self.cache[tile_key] = {
            **data,
            'timestamp': time.time()
        }
    
    def clear_expired(self):
        current_time = time.time()
        expired_keys = [
            key for key, value in self.cache.items()
            if current_time - value['timestamp'] > 300  # 5 minutes cache
        ]
        for key in expired_keys:
            del self.cache[key]

class TileProcessor:
    def __init__(self):
        self.cache = TileCache()
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        
    async def download_tile(self, session: aiohttp.ClientSession, tile_id: str, tile_num: str) -> bytes:
        """Download tile image with rate limiting"""
        async with self.semaphore:
            url = f"https://backend.wplace.live/files/s0/tiles/{tile_id}/{tile_num}.png"
            try:
                logger.debug(f"Downloading tile {tile_id}/{tile_num}...")
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.read()
                        logger.debug(f"Downloaded {len(data)} bytes for tile {tile_id}/{tile_num}")
                        return data
                    else:
                        logger.warning(f"Failed to download {url}: {response.status}")
                        return None
            except Exception as e:
                logger.error(f"Error downloading {url}: {e}")
                return None
    
    async def save_file(self, filepath: str, data: bytes):
        """Save file asynchronously"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(data)
    
    def process_tile_diff(self, base_path: str, blueprint_path: str) -> Tuple[bool, List, int]:
        """Process tile differences with blueprint overlay and pink highlighting"""
        try:
            base_img = Image.open(base_path).convert('RGBA')
            blueprint_img = Image.open(blueprint_path).convert('RGBA')
            
            # Convert to arrays for faster processing
            base_array = base_img.load()
            blueprint_array = blueprint_img.load()
            
            width, height = base_img.size
            diff_pixels = []
            missing_count = 0
            xmin, xmax = 999, 0
            ymin, ymax = 999, 0
            
            # First pass: find differences and boundaries
            for x in range(width):
                for y in range(height):
                    bp_pixel = blueprint_array[x, y]
                    base_pixel = base_array[x, y]
                    
                    # Check if blueprint pixel is not transparent and differs from base
                    if bp_pixel[3] > 0 and bp_pixel != base_pixel:
                        missing_count += 1
                        diff_pixels.append(((x, y), (bp_pixel[0], bp_pixel[1], bp_pixel[2], 255)))
                        
                        # Track boundaries for pink overlay
                        xmin = min(x, xmin)
                        xmax = max(x, xmax)
                        ymin = min(y, ymin)
                        ymax = max(y, ymax)
            
            if diff_pixels:
                # Apply pink overlay around changed regions (like original code)
                if xmin != 999 and ymin != 999:  # Only if we found differences
                    for x in range(max(0, xmin-4), min(width, xmax+5)):
                        for y in range(max(0, ymin-4), min(height, ymax+5)):
                            base_array[x, y] = (255, 0, 255, 80)  # Pink with transparency
                
                # Apply blueprint pixels where they differ
                for (x, y), color in diff_pixels:
                    base_array[x, y] = color
                
                # Save modified image
                base_img.save(base_path, 'PNG', optimize=True)
                
            return len(diff_pixels) > 0, diff_pixels, missing_count
            
        except Exception as e:
            logger.error(f"Error processing tile diff: {e}")
            return False, [], 0
    
    async def process_tile(self, session: aiohttp.ClientSession, tile_id: str, tile_num: str) -> Dict:
        """Process a single tile asynchronously"""
        tile_key = f"{tile_id}_{tile_num}"
        
        # Check cache first
        if self.cache.is_valid(tile_key):
            return self.cache.get(tile_key)
        
        base_path = f'files/s0/tiles/{tile_id}/{tile_num}.png'
        blueprint_path = f'blueprints/{tile_id}/{tile_num}blueprint.png'
        
        # Download current tile
        img_data = await self.download_tile(session, tile_id, tile_num)
        if img_data is None:
            return {'error': 'Download failed', 'missing_pixels': 0}
        
        # Save current tile
        await self.save_file(base_path, img_data)
        
        # Create blueprint if it doesn't exist
        if not os.path.isfile(blueprint_path):
            os.makedirs(os.path.dirname(blueprint_path), exist_ok=True)
            await self.save_file(blueprint_path, img_data)
        
        # Process differences
        has_diff, diff_pixels, missing_count = self.process_tile_diff(base_path, blueprint_path)
        
        logger.info(f"Tile {tile_id}/{tile_num}: {missing_count} missing pixels, {len(diff_pixels)} differences. Estimated needed time:{diff_pixels/NUMBER_OF_ACCOUNTS/120}h")
        
        result = {
            'tile_id': tile_id,
            'tile_num': tile_num,
            'missing_pixels': missing_count,
            'has_differences': has_diff,
            'diff_count': len(diff_pixels)
        }
        
        # Cache the result
        self.cache.set(tile_key, result)
        
        return result
    
    async def update_all_tiles(self, tiles_config: List[List[str]], force_fresh: bool = False) -> List[Dict]:
        """Update all tiles concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for tile in tiles_config:
                if force_fresh:
                    # Force fresh download for background updates
                    task = self.force_update_tile_fresh(session, tile[0], tile[1])
                else:
                    task = self.process_tile(session, tile[0], tile[1])
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out errors and calculate totals
            valid_results = [r for r in results if not isinstance(r, Exception)]
            total_missing = sum(r.get('missing_pixels', 0) for r in valid_results)
            
            logger.info(f"Updated {len(valid_results)} tiles. Total missing pixels: {total_missing}")
            return valid_results
    
    async def force_update_tile_fresh(self, session: aiohttp.ClientSession, tile_id: str, tile_num: str) -> Dict:
        """Force fresh download and processing of a tile (bypasses cache)"""
        try:
            tile_key = f"{tile_id}_{tile_num}"
            
            base_path = f'files/s0/tiles/{tile_id}/{tile_num}.png'
            blueprint_path = f'blueprints/{tile_id}/{tile_num}blueprint.png'
            
            # Always download fresh tile
            img_data = await self.download_tile(session, tile_id, tile_num)
            if img_data is None:
                return {'error': 'Download failed', 'missing_pixels': 0}
            
            # Save current tile
            await self.save_file(base_path, img_data)
            
            # Create blueprint if it doesn't exist
            if not os.path.isfile(blueprint_path):
                os.makedirs(os.path.dirname(blueprint_path), exist_ok=True)
                await self.save_file(blueprint_path, img_data)
            
            # Process differences
            has_diff, diff_pixels, missing_count = self.process_tile_diff(base_path, blueprint_path)
            estimate=len(diff_pixels)/NUMBER_OF_ACCOUNTS/120
            now = datetime.now()
            logger.info(f"Fresh update - Tile {tile_id}/{tile_num}: {missing_count} missing pixels, {len(diff_pixels)} differences. Estimated needed time:{estimate}h. Estimated datetime at finish: {now + timedelta(hours=estimate)}")
            
            result = {
                'tile_id': tile_id,
                'tile_num': tile_num,
                'missing_pixels': missing_count,
                'has_differences': has_diff,
                'diff_count': len(diff_pixels)
            }
            
            # Update cache with fresh result
            self.cache.set(tile_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in force_update_tile_fresh for {tile_id}/{tile_num}: {e}")
            return {'error': str(e), 'missing_pixels': 0}
    
    async def force_update_tile(self, tile_path: str) -> bool:
        """Force update a specific tile by path"""
        try:
            # Extract tile_id and tile_num from path like "s0/tiles/1089/705.png"
            path_parts = tile_path.split('/')
            if len(path_parts) >= 4:
                tile_id = path_parts[2]
                tile_num = path_parts[3].replace('.png', '')
                
                logger.info(f"Force updating tile {tile_id}/{tile_num}")
                
                async with aiohttp.ClientSession() as session:
                    result = await self.process_tile(session, tile_id, tile_num)
                    return result.get('error') is None
            else:
                logger.error(f"Invalid tile path format: {tile_path}")
                return False
        except Exception as e:
            logger.error(f"Error force updating tile {tile_path}: {e}")
            return False



class WebServer:
    def __init__(self):
        self.processor = TileProcessor()
        self.app = web.Application()
        self.last_update_time = 0
        self.next_update_time = 0
        self.update_count = 0
        self.setup_routes()
        
    def setup_routes(self):
        """Setup web routes"""
        self.app.router.add_get('/', self.index_handler)
        self.app.router.add_get('/config.json', self.config_handler)
        self.app.router.add_get('/files/{path:.*}', self.files_handler)
        self.app.router.add_post('/update', self.update_handler)
        self.app.router.add_get('/status', self.status_handler)
        self.app.router.add_options('/{path:.*}', self.options_handler)
        
        # CORS headers are now added directly to each handler
    
    async def index_handler(self, request):
        """Serve index page"""
        response = web.Response(text="wPlace Overlay Server Running", content_type='text/plain')
        return add_cors_headers(response)
    
    async def config_handler(self, request):
        """Serve config.json"""
        try:
            async with aiofiles.open('config.json', 'r') as f:
                content = await f.read()
            response = web.Response(text=content, content_type='application/json')
            return add_cors_headers(response)
        except Exception as e:
            logger.error(f"Error reading config: {e}")
            response = web.Response(text='{"error": "Config not found"}', status=500)
            return add_cors_headers(response)
    
    async def files_handler(self, request):
        """Serve static files with caching and validation"""
        path = request.match_info['path']
        file_path = f'files/{path}'
        
        # Log file requests for debugging
        logger.debug(f"Serving file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            response = web.Response(status=404)
            return add_cors_headers(response)
        
        try:
            # Verify file is valid and not corrupted
            file_size = os.path.getsize(file_path)
            if file_size < 100:  # PNG files should be at least 100 bytes
                logger.warning(f"File {file_path} seems corrupted (size: {file_size} bytes)")
                # Try to regenerate the tile
                await self.processor.force_update_tile(path)
                # Check if regeneration helped
                if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
                    logger.info(f"File {file_path} regenerated successfully")
                else:
                    return web.Response(status=500, text="Tile processing failed")
            
            # Additional PNG validation
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(8)
                    if header != b'\x89PNG\r\n\x1a\n':
                        logger.warning(f"File {file_path} is not a valid PNG")
                        # Try to regenerate
                        await self.processor.force_update_tile(path)
                        if os.path.exists(file_path):
                            with open(file_path, 'rb') as f2:
                                header2 = f2.read(8)
                                if header2 != b'\x89PNG\r\n\x1a\n':
                                    return web.Response(status=500, text="Invalid PNG file")
                        else:
                            return web.Response(status=500, text="Tile regeneration failed")
            except Exception as e:
                logger.error(f"Error validating PNG {file_path}: {e}")
                return web.Response(status=500, text="File validation failed")
            
            # Add cache headers and CORS
            response = web.FileResponse(file_path)
            
            # Very short cache for tile images to ensure fresh updates
            # Since we update every 10 seconds, cache for only 5 seconds
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            response.headers['ETag'] = f'"{os.path.getmtime(file_path)}"'
            response.headers['Last-Modified'] = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime(os.path.getmtime(file_path)))
            
            # Log successful file serve
            file_size = os.path.getsize(file_path)
            file_mtime = os.path.getmtime(file_path)
            logger.info(f"Served {file_path} (size: {file_size}, mtime: {file_mtime})")
            
            return add_cors_headers(response)
            
        except Exception as e:
            logger.error(f"Error serving file {file_path}: {e}")
            response = web.Response(status=500, text="Internal server error")
            return add_cors_headers(response)
    
    async def status_handler(self, request):
        """Handle status requests"""
        try:
            current_time = time.time()
            time_since_last = current_time - self.last_update_time if self.last_update_time > 0 else None
            time_until_next = self.next_update_time - current_time if self.next_update_time > current_time else 0
            
            status_data = {
                'server_time': current_time,
                'last_update': self.last_update_time,
                'time_since_last_update': time_since_last,
                'next_update_in': time_until_next,
                'update_count': self.update_count,
                'update_interval': UPDATE_INTERVAL,
                'cache_size': len(self.processor.cache.cache),
                'uptime': current_time - self.start_time if hasattr(self, 'start_time') else None
            }
            
            response = web.json_response(status_data)
            return add_cors_headers(response)
        except Exception as e:
            logger.error(f"Status error: {e}")
            response = web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
            return add_cors_headers(response)
    
    async def update_handler(self, request):
        """Handle manual update requests"""
        try:
            with open('config.json') as f:
                tiles_config = json.load(f)
            
            results = await self.processor.update_all_tiles(tiles_config)
            response = web.json_response({
                'success': True,
                'results': results,
                'timestamp': time.time()
            })
            return add_cors_headers(response)
        except Exception as e:
            logger.error(f"Update error: {e}")
            response = web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
            return add_cors_headers(response)
    
    async def options_handler(self, request):
        """Handle OPTIONS requests for CORS preflight"""
        response = web.Response()
        response.headers['Access-Control-Max-Age'] = '86400'  # 24 hours
        return add_cors_headers(response)

async def background_updater(server: WebServer):
    """Background task to periodically update tiles"""
    logger.info(f"Background updater started - will update every {UPDATE_INTERVAL} seconds")
    
    # Wait a bit before first update to let server fully start
    await asyncio.sleep(10)
    
    while True:
        try:
            start_time = time.time()
            logger.info("Starting background tile update...")
            
            with open('config.json') as f:
                tiles_config = json.load(f)
            
            results = await server.processor.update_all_tiles(tiles_config, force_fresh=True)
            server.processor.cache.clear_expired()
            
            # Update server tracking
            server.last_update_time = time.time()
            server.next_update_time = server.last_update_time + UPDATE_INTERVAL
            server.update_count += 1
            
            update_duration = time.time() - start_time
            
            # Log summary of what happened
            total_missing = sum(r.get('missing_pixels', 0) for r in results if r.get('error') is None)
            logger.info(f"Background update completed in {update_duration:.2f}s (total updates: {server.update_count})")
            logger.info(f"Processed {len(results)} tiles, total missing pixels: {total_missing}")
            
            # Calculate wait time to maintain consistent intervals
            wait_time = max(0, UPDATE_INTERVAL - update_duration)
            logger.info(f"Waiting {wait_time:.1f}s until next update (target: every {UPDATE_INTERVAL}s)")
            await asyncio.sleep(wait_time)
            
        except Exception as e:
            logger.error(f"Background update error: {e}")
            logger.info("Retrying in 30 seconds...")
            await asyncio.sleep(30)

async def main():
    """Main application entry point"""
    try:
        # Load configuration
        config = load_config()
        logger.info(f"Loaded configuration: update_interval={config['update_interval']}s, port={config['port']}")
        
        logger.info("Initializing WebServer...")
        server = WebServer()
        logger.info("WebServer initialized successfully")
        
        # Start background updater
        logger.info("Starting background updater...")
        server.start_time = time.time()
        asyncio.create_task(background_updater(server))
        logger.info("Background updater started")
        
        # Start web server
        logger.info("Setting up web server...")
        runner = web.AppRunner(server.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', config['port'])
        
        logger.info(f"Starting server on port {config['port']}")
        await site.start()
        logger.info("Server started successfully")
        
        # Keep server running
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await runner.cleanup()
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting wPlace Overlay Server...")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
