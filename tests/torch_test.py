import torch
import torch.nn.functional as F
import time

print("=" * 50)
print("PyTorch & CUDA Diagnostics")
print("=" * 50)

# Basic version info
print(f"\n[INFO] PyTorch version: {torch.__version__}")
print(f"[INFO] CUDA version:    {torch.version.cuda}")
print(f"[INFO] cuDNN version:   {torch.backends.cudnn.version()}")
print(f"[INFO] CUDA available:  {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available! Exiting.")
    exit(1)

# Device info
print(f"[INFO] CUDA device:     {torch.cuda.get_device_name(0)}")
print(f"[INFO] Device count:    {torch.cuda.device_count()}")
print(f"[INFO] Current device:  {torch.cuda.current_device()}")

# Memory info
total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"[INFO] GPU memory:      {total_mem:.2f} GB")

print("\n" + "=" * 50)
print("Running Tests...")
print("=" * 50)

# Test 1: Basic tensor operations on GPU
print("\n[TEST 1] Basic tensor operations on GPU...")
try:
    a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    b = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    c = a + b
    assert c.device.type == 'cuda'
    assert torch.allclose(c, torch.tensor([5.0, 7.0, 9.0], device='cuda'))
    print("[PASS] Basic tensor ops work correctly")
except Exception as e:
    print(f"[FAIL] {e}")

# Test 2: CPU <-> GPU transfer
print("\n[TEST 2] CPU <-> GPU data transfer...")
try:
    cpu_tensor = torch.randn(100, 100)
    gpu_tensor = cpu_tensor.cuda()
    back_to_cpu = gpu_tensor.cpu()
    assert torch.allclose(cpu_tensor, back_to_cpu)
    print("[PASS] CPU <-> GPU transfer works")
except Exception as e:
    print(f"[FAIL] {e}")

# Test 3: Convolution (cuDNN)
print("\n[TEST 3] Convolution (tests cuDNN)...")
try:
    x = torch.randn(1, 3, 64, 64, device='cuda')
    w = torch.randn(8, 3, 3, 3, device='cuda')
    y = F.conv2d(x, w)
    assert y.shape == (1, 8, 62, 62)
    print(f"[PASS] Conv2d output shape: {y.shape}")
except Exception as e:
    print(f"[FAIL] {e}")

# Test 4: Matrix multiplication
print("\n[TEST 4] Matrix multiplication (cuBLAS)...")
try:
    m1 = torch.randn(256, 256, device='cuda')
    m2 = torch.randn(256, 256, device='cuda')
    result = torch.mm(m1, m2)
    assert result.shape == (256, 256)
    print("[PASS] Matrix multiplication works")
except Exception as e:
    print(f"[FAIL] {e}")

# Test 5: Gradient computation
print("\n[TEST 5] Autograd (gradient computation)...")
try:
    x = torch.randn(3, 3, device='cuda', requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    expected_grad = 2 * x
    assert torch.allclose(x.grad, expected_grad.detach())
    print("[PASS] Autograd works correctly")
except Exception as e:
    print(f"[FAIL] {e}")

# Test 6: Simple neural network forward pass
print("\n[TEST 6] Neural network forward pass...")
try:
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    ).cuda()
    input_tensor = torch.randn(8, 64, device='cuda')
    output = model(input_tensor)
    assert output.shape == (8, 10)
    print(f"[PASS] NN forward pass output shape: {output.shape}")
except Exception as e:
    print(f"[FAIL] {e}")

# Test 7: Performance benchmark
print("\n[TEST 7] Performance benchmark (matmul 1024x1024)...")
try:
    torch.cuda.synchronize()
    a = torch.randn(1024, 1024, device='cuda')
    b = torch.randn(1024, 1024, device='cuda')
    
    # Warmup
    for _ in range(10):
        torch.mm(a, b)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"[PASS] 100x matmul (1024x1024): {elapsed*1000:.2f}ms ({elapsed*10:.2f}ms per op)")
except Exception as e:
    print(f"[FAIL] {e}")

# Test 8: Half precision (FP16)
print("\n[TEST 8] Half precision (FP16) support...")
try:
    x_fp16 = torch.randn(100, 100, device='cuda', dtype=torch.float16)
    y_fp16 = torch.mm(x_fp16, x_fp16.t())
    assert y_fp16.dtype == torch.float16
    print("[PASS] FP16 operations work")
except Exception as e:
    print(f"[FAIL] {e}")

print("\n" + "=" * 50)
print("All tests completed!")
print("=" * 50)
