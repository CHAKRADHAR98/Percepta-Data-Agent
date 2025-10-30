"""
Pinata IPFS Handler for Bimanual Data Agent
"""
import os
import json
import time
import requests
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from utils import extract_ipfs_cid, format_file_size
from config import Config


class PinataHandler:
    """Handle IPFS operations via Pinata"""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api.pinata.cloud"
    
    def download_video(self, ipfs_url: str, output_dir: Path) -> Optional[Path]:
        """
        Download video from IPFS with multiple gateway fallbacks
        
        Args:
            ipfs_url: IPFS URL (any format)
            output_dir: Where to save the video
            
        Returns:
            Path to downloaded video or None
        """
        print("\n" + "="*70)
        print("📥 DOWNLOADING VIDEO FROM IPFS")
        print("="*70)
        
        # Extract CID
        cid = extract_ipfs_cid(ipfs_url)
        if not cid:
            print(f"❌ Invalid IPFS URL: {ipfs_url}")
            return None
        
        # Try multiple IPFS gateways (in order of preference)
        gateways = [
            f"https://gateway.pinata.cloud/ipfs/{cid}",
            f"https://ipfs.io/ipfs/{cid}",
            f"https://cloudflare-ipfs.com/ipfs/{cid}",
            f"https://dweb.link/ipfs/{cid}",
            f"https://gateway.ipfs.io/ipfs/{cid}",
        ]
        
        last_error = None
        
        for gateway_url in gateways:
            try:
                gateway_name = gateway_url.split('/')[2]
                print(f"\n🔗 Trying: {gateway_name}")
                print(f"   URL: {gateway_url}")
                
                # Make request
                print("   🌐 Connecting...")
                response = requests.get(gateway_url, stream=True, timeout=60)
                response.raise_for_status()
                response.raise_for_status()
                
                # Get file info
                total_size = int(response.headers.get('content-length', 0))
                filename = f"ipfs_{cid[:8]}.mp4"
                output_path = output_dir / filename
                
                print(f"   📁 Filename: {filename}")
                if total_size > 0:
                    print(f"   📦 Size: {format_file_size(total_size)}")
                
                # Download with progress
                print(f"   ⬇️  Downloading...")
                downloaded = 0
                chunk_size = 8192
                start_time = time.time()
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                speed = downloaded / (time.time() - start_time) / 1024 / 1024
                                
                                bar_length = 40
                                filled = int(bar_length * downloaded / total_size)
                                bar = '█' * filled + '░' * (bar_length - filled)
                                
                                print(f"\r      [{bar}] {percent:.1f}% | {speed:.2f} MB/s", end='')
                
                print("\n")
                
                # Validate
                file_size = os.path.getsize(output_path)
                if file_size == 0:
                    print("   ❌ Downloaded file is empty")
                    os.remove(output_path)
                    last_error = "Empty file"
                    continue
                
                print(f"   ✅ Download successful from {gateway_name}!")
                print(f"   💾 Saved to: {output_path}")
                print("="*70)
                
                return output_path
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"   ⚠️  Rate limited (429) - trying next gateway...")
                    last_error = f"Rate limited on {gateway_name}"
                else:
                    print(f"   ⚠️  HTTP {e.response.status_code} - trying next gateway...")
                    last_error = str(e)
                continue
                
            except requests.exceptions.Timeout:
                print(f"   ⚠️  Timeout (60s) - trying next gateway...")
                last_error = f"Timeout on {gateway_name}"
                continue
                
            except Exception as e:
                print(f"   ⚠️  Error: {e} - trying next gateway...")
                last_error = str(e)
                continue
        
        # All gateways failed
        print(f"\n❌ All IPFS gateways failed!")
        print(f"   Last error: {last_error}")
        print(f"   CID: {cid}")
        print("\n💡 Tips:")
        print("   • Wait a few minutes and try again")
        print("   • Check if the CID is valid")
        print("   • Try accessing directly: https://ipfs.io/ipfs/{cid}")
        print("="*70)
        return None
    
    def upload_dataset(self, task_id: str, base_name: str, 
                      dataset_dir: Path = Config.DATASET_DIR) -> Optional[Dict[str, str]]:
        """
        Upload complete dataset folder to IPFS using Pinata API directly
        
        Args:
            task_id: Task identifier (e.g., "opening_bottle")
            base_name: Base name of video file (e.g., "ipfs_bafybeif")
            dataset_dir: Dataset root directory
            
        Returns:
            Dictionary with IPFS hash and manifest or None
        """
        print("\n" + "="*70)
        print("📤 UPLOADING DATASET TO IPFS")
        print("="*70)
        
        try:
            # Create temporary upload folder
            upload_folder_name = f"{task_id}_{base_name}_dataset"
            temp_upload_dir = Config.TEMP_DIR / upload_folder_name
            
            # Clean up if exists
            if temp_upload_dir.exists():
                shutil.rmtree(temp_upload_dir)
            temp_upload_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n📁 Preparing dataset folder: {upload_folder_name}")
            
            # Collect all files to upload
            files_to_copy = {}
            
            # 1. Annotated video (if exists)
            video_path = dataset_dir / "videos" / "annotated" / f"{task_id}_{base_name}_annotated.mp4"
            if video_path.exists():
                files_to_copy['video_annotated.mp4'] = video_path
                print(f"   ✓ Found: annotated video")
            else:
                print(f"   ⚠️  Skipping: annotated video (not found)")
            
            # 2. Original video (optional)
            orig_video_path = dataset_dir / "videos" / "from_ipfs" / f"{base_name}.mp4"
            if orig_video_path.exists():
                files_to_copy['video_original.mp4'] = orig_video_path
                print(f"   ✓ Found: original video")
            
            # 3. Trajectory data (CRITICAL)
            npy_path = dataset_dir / "annotations" / "hand_trajectories" / f"{task_id}_{base_name}_actions.npy"
            if npy_path.exists():
                files_to_copy['actions.npy'] = npy_path
                print(f"   ✓ Found: trajectory data (.npy)")
            else:
                print(f"   ❌ Missing critical file: actions.npy")
                return None
            
            # 4. Metadata files
            metadata_files = {
                'actions.json': dataset_dir / "annotations" / "hand_trajectories" / f"{task_id}_{base_name}_actions.json",
                'actions.csv': dataset_dir / "annotations" / "hand_trajectories" / f"{task_id}_{base_name}_actions.csv",
                'quality_report.json': dataset_dir / "annotations" / "quality_reports" / f"{task_id}_{base_name}_quality.json",
                'metadata.json': dataset_dir / "metadata" / f"{task_id}_{base_name}_metadata.json"
            }
            
            for target_name, source_path in metadata_files.items():
                if source_path.exists():
                    files_to_copy[target_name] = source_path
                    print(f"   ✓ Found: {target_name}")
            
            # 5. Trajectory plot (if exists)
            plot_path = Config.OUTPUT_DIR / f"{task_id}_{base_name}_trajectories.png"
            if plot_path.exists():
                files_to_copy['trajectories.png'] = plot_path
                print(f"   ✓ Found: trajectory plot")
            
            # Copy all files to temp upload folder
            print(f"\n📦 Copying files to upload folder...")
            for target_name, source_path in files_to_copy.items():
                dest_path = temp_upload_dir / target_name
                shutil.copy2(source_path, dest_path)
                print(f"   ✓ {target_name}")
            
            # Upload using multipart form data (Pinata API)
            print(f"\n☁️  Uploading folder to IPFS...")
            print(f"   📂 Folder: {upload_folder_name}")
            print(f"   📄 Files: {len(files_to_copy)}")
            
            # Prepare multipart form data with proper paths
            files_payload = []
            for filename in files_to_copy.keys():
                file_path = temp_upload_dir / filename
                # Use relative path: folder_name/filename
                relative_path = f"{upload_folder_name}/{filename}"
                files_payload.append(
                    ('file', (relative_path, open(file_path, 'rb'), 'application/octet-stream'))
                )
            
            # Prepare metadata
            metadata = {
                'name': upload_folder_name,
                'keyvalues': {
                    'type': 'bimanual_dataset',
                    'task_id': task_id,
                    'format': 'folder'
                }
            }
            
            # Upload to Pinata
            headers = {
                'pinata_api_key': self.api_key,
                'pinata_secret_api_key': self.secret_key
            }
            
            data = {
                'pinataMetadata': json.dumps(metadata),
                'pinataOptions': json.dumps({'wrapWithDirectory': False})
            }
            
            response = requests.post(
                f"{self.base_url}/pinning/pinFileToIPFS",
                headers=headers,
                files=files_payload,
                data=data,
                timeout=300  # 5 minute timeout for large files
            )
            
            # Close all file handles
            for _, (_, file_handle, _) in files_payload:
                file_handle.close()
            
            if response.status_code != 200:
                print(f"   ❌ Upload failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
            
            result = response.json()
            folder_hash = result.get('IpfsHash')
            
            if not folder_hash:
                print(f"   ❌ Upload failed: couldn't get IPFS hash")
                return None
            
            print(f"\n   ✅ Uploaded successfully!")
            print(f"   🔗 IPFS Hash: {folder_hash}")
            
            # Create manifest
            manifest = {
                'dataset_name': upload_folder_name,
                'task_id': task_id,
                'version': '1.0',
                'folder_hash': folder_hash,
                'folder_url': f"https://gateway.pinata.cloud/ipfs/{folder_hash}/{upload_folder_name}",
                'files': {
                    filename: f"https://gateway.pinata.cloud/ipfs/{folder_hash}/{upload_folder_name}/{filename}"
                    for filename in files_to_copy.keys()
                },
                'upload_timestamp': time.time()
            }
            
            # Save manifest locally
            manifest_path = dataset_dir / "metadata" / f"{task_id}_{base_name}_ipfs_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            print(f"\n   💾 Manifest saved: {manifest_path.name}")
            
            # Clean up temp folder
            shutil.rmtree(temp_upload_dir)
            
            print("\n" + "="*70)
            print("✅ UPLOAD COMPLETE")
            print("="*70)
            print(f"\n📦 **Dataset Folder:** `{folder_hash}`")
            print(f"🌐 **Browse:** https://gateway.pinata.cloud/ipfs/{folder_hash}/{upload_folder_name}")
            print(f"\n📄 **Access Files:**")
            for filename in ['actions.npy', 'actions.json', 'quality_report.json']:
                if filename in files_to_copy:
                    print(f"   • {filename}: https://gateway.pinata.cloud/ipfs/{folder_hash}/{upload_folder_name}/{filename}")
            print("="*70)
            
            return manifest
            
        except Exception as e:
            print(f"\n❌ Upload error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_manifest(self, manifest_cid: str) -> Optional[Dict]:
        """Download and parse manifest from IPFS"""
        try:
            url = f"https://gateway.pinata.cloud/ipfs/{manifest_cid}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching manifest: {e}")
            return None