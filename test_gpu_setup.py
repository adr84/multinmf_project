import platform
import subprocess
import sys
import os
from datetime import datetime

def run_command(command):
    """Run a system command safely"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return -1, "", str(e)

def check_macos_version():
    """Check macOS version for MPS compatibility"""
    print("macOS Version Check:")
    print("-" * 20)
    
    code, output, error = run_command("sw_vers")
    if code == 0:
        lines = output.split('\n')
        version_info = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                version_info[key.strip()] = value.strip()
        
        product_name = version_info.get('ProductName', 'Unknown')
        version = version_info.get('ProductVersion', 'Unknown')
        build = version_info.get('BuildVersion', 'Unknown')
        
        print(f"Product: {product_name}")
        print(f"Version: {version}")
        print(f"Build: {build}")
        
        # Check if version supports MPS (macOS 12.3+)
        try:
            major, minor, patch = version.split('.')
            major, minor = int(major), int(minor)
            
            if major > 12 or (major == 12 and minor >= 3):
                print("✓ macOS version supports MPS")
                return True
            else:
                print("✗ macOS 12.3+ required for MPS support")
                print(f"  Current: {version}, Required: 12.3+")
                return False
        except:
            print("? Unable to parse version number")
            return False
    else:
        print(f"✗ Could not get macOS version: {error}")
        return False

def check_hardware():
    """Check Apple hardware details"""
    print("\nHardware Information:")
    print("-" * 22)
    
    # Get chip information
    code, output, error = run_command("system_profiler SPHardwareDataType")
    if code == 0:
        lines = output.split('\n')
        for line in lines:
            if 'Chip:' in line:
                chip = line.split(':', 1)[1].strip()
                print(f"Chip: {chip}")
                
                if 'M1' in chip or 'M2' in chip or 'M3' in chip:
                    print("✓ Apple Silicon detected - MPS support available")
                    apple_silicon = True
                else:
                    print("✗ Not Apple Silicon - MPS not available")
                    apple_silicon = False
                    
            elif 'Memory:' in line:
                memory = line.split(':', 1)[1].strip()
                print(f"Memory: {memory}")
                
        return apple_silicon
    else:
        print("? Could not get hardware information")
        return False

def check_python_environment():
    """Check Python and package installations"""
    print("\nPython Environment:")
    print("-" * 19)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Virtual environment detected")
    else:
        print("ℹ Using system Python (consider using virtual environment)")

def check_pytorch_mps():
    """Check PyTorch installation and MPS support"""
    print("\nPyTorch MPS Check:")
    print("-" * 18)
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        
        # Check MPS availability
        if hasattr(torch.backends, 'mps'):
            mps_available = torch.backends.mps.is_available()
            mps_built = torch.backends.mps.is_built()
            
            print(f"MPS built: {mps_built}")
            print(f"MPS available: {mps_available}")
            
            if mps_available and mps_built:
                print("✓ MPS GPU acceleration ready!")
                
                # Test basic MPS operations
                try:
                    device = torch.device("mps")
                    x = torch.randn(100, 100, device=device)
                    y = torch.mm(x, x.T)
                    print("✓ Basic MPS operations working")
                    
                    # Clean up
                    del x, y
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                    
                    return True
                except Exception as e:
                    print(f"✗ MPS operations failed: {e}")
                    return False
            else:
                print("✗ MPS not available")
                if not mps_built:
                    print("  PyTorch was not built with MPS support")
                    print("  Try: pip install torch torchvision torchaudio")
                return False
        else:
            print("✗ PyTorch version too old for MPS")
            print("  Update PyTorch: pip install --upgrade torch")
            return False
            
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch torchvision torchaudio")
        return False

def check_required_packages():
    """Check other required packages"""
    print("\nRequired Packages Check:")
    print("-" * 25)
    
    packages = {
        'numpy': 'NumPy',
        'scipy': 'SciPy', 
        'matplotlib': 'Matplotlib',
        'h5py': 'HDF5 for Python (optional)'
    }
    
    missing = []
    
    for package, name in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name} {version}")
        except ImportError:
            print(f"✗ {name} not installed")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def run_performance_test():
    """Run a quick performance test"""
    print("\nPerformance Test:")
    print("-" * 16)
    
    try:
        import torch
        import time
        
        if not torch.backends.mps.is_available():
            print("Skipping - MPS not available")
            return
        
        # Test matrix sizes
        sizes = [500, 1000, 2000]
        
        for size in sizes:
            print(f"\nMatrix size: {size}x{size}")
            
            # CPU test
            x_cpu = torch.randn(size, size)
            start_time = time.time()
            y_cpu = torch.mm(x_cpu, x_cpu.T)
            cpu_time = time.time() - start_time
            print(f"  CPU: {cpu_time:.3f} seconds")
            
            # MPS test
            x_mps = x_cpu.to('mps')
            start_time = time.time()
            y_mps = torch.mm(x_mps, x_mps.T)
            torch.mps.synchronize()  # Wait for GPU to finish
            mps_time = time.time() - start_time
            print(f"  MPS: {mps_time:.3f} seconds")
            
            speedup = cpu_time / mps_time
            print(f"  Speedup: {speedup:.2f}x")
            
            # Cleanup
            del x_cpu, y_cpu, x_mps, y_mps
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
                
    except Exception as e:
        print(f"Performance test failed: {e}")

def generate_recommendations():
    """Generate setup recommendations"""
    print("\n" + "="*50)
    print("SETUP RECOMMENDATIONS")
    print("="*50)
    
    # Check all components
    macos_ok = check_macos_version()
    hardware_ok = check_hardware()
    python_ok = check_python_environment()
    pytorch_ok = check_pytorch_mps()
    packages_ok = check_required_packages()
    
    print(f"\nSetup Status:")
    print(f"{'macOS Version:':<15} {'✓ Ready' if macos_ok else '✗ Needs update'}")
    print(f"{'Hardware:':<15} {'✓ Apple Silicon' if hardware_ok else '✗ Not compatible'}")
    print(f"{'Python:':<15} {'✓ Ready' if python_ok else '✗ Issues detected'}")
    print(f"{'PyTorch MPS:':<15} {'✓ Ready' if pytorch_ok else '✗ Needs setup'}")
    print(f"{'Packages:':<15} {'✓ Ready' if packages_ok else '✗ Missing packages'}")
    
    if all([macos_ok, hardware_ok, pytorch_ok, packages_ok]):
        print(f"\n🎉 Your M1 system is ready for GPU-accelerated MultiNMF!")
        print(f"\nNext steps:")
        print(f"1. Download the M1-optimized MultiNMF scripts")
        print(f"2. Run the performance test")
        print(f"3. Execute your MultiNMF analysis")
        
        # Run performance test if everything is ready
        run_performance_test()
        
    else:
        print(f"\n⚠️ Setup needs attention:")
        
        if not macos_ok:
            print(f"• Update macOS to 12.3 or later for MPS support")
        if not hardware_ok:
            print(f"• Apple Silicon required for MPS acceleration")
        if not pytorch_ok:
            print(f"• Install/update PyTorch: pip install torch torchvision torchaudio")
        if not packages_ok:
            print(f"• Install missing Python packages")
    
    return all([macos_ok, hardware_ok, pytorch_ok, packages_ok])

def main():
    """Main hardware check function"""
    print("Apple M1 Hardware Check for MultiNMF GPU Acceleration")
    print("="*55)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run comprehensive check
    ready = generate_recommendations()
    
    print(f"\n{'='*55}")
    if ready:
        print("✅ System ready for M1 GPU acceleration!")
    else:
        print("❌ Setup incomplete - see recommendations above")
    
    return ready

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
