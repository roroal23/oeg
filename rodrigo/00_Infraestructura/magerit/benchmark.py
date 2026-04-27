import torch
import time

SIZE = 32
ITERATIONS = 100

def benchmark(device):
    print(f"===Benchmarking {device}===")
    #Creamos 2 matrices cuadradas
    a = torch.randn(SIZE, SIZE, device = 'cuda', dtype = torch.float16)
    b = torch.randn(SIZE, SIZE, device = 'cuda', dtype = torch.float16)

    #"Calentamiento"
    for _ in range(10):
        torch.matmul(a, b)

    torch.cuda.synchronize() #Esperamos a que todos los nucleos acaben
    start = time.time()

    for _ in range(ITERATIONS):
        torch.matmul(a, b)

    torch.cuda.synchronize()  #Esperamos a que todos los nucleos acaben
    end = time.time()

    avg_duration = (end - start) / ITERATIONS
    tflops = (2 * SIZE**3) / (avg_duration * 1e12)
    print(f"Tiempo medio: {avg_duration*1000:.2f} ms")
    print(f"Rendimiento estimado: {tflops:.2f} Tflops")

    if __name__ == "__main__":
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            benchmark(device)
        else:
            print("GPU no disponible")
