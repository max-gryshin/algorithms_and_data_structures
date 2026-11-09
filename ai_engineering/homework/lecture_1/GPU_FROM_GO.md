# Обучение на GPU из Go

Да, Go может работать с GPU через cgo. Но на Apple Silicon (M5 Pro) CUDA нет: она работает только на картах NVIDIA. Ниже варианты по порядку пользы.

## 1. Accelerate (`cblas_sgemm`) через cgo: самое дешёвое и самое выгодное

Формально это CPU, но matmul там считается на матричных блоках Apple (AMX/SME). Замер на M5 Pro при 1024×1024:

| Ядро | GFLOP/s |
|---|---|
| `MatMulParallel` из `tensor/matmul.go` | 77 |
| Accelerate `cblas_sgemm` | **1124** |

Получается примерно в 15 раз быстрее, и для этого нужно около 10 строк:

```go
/*
#cgo CFLAGS: -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"
import "unsafe"

// out[n×p] = a[n×k] · b[k×p], row-major
func MatMulAccelerate(a, b, out []float32, n, k, p int) {
	C.cblas_sgemm(C.CblasRowMajor, C.CblasNoTrans, C.CblasNoTrans,
		C.__LAPACK_int(n), C.__LAPACK_int(p), C.__LAPACK_int(k),
		1, (*C.float)(unsafe.Pointer(&a[0])), C.__LAPACK_int(k),
		(*C.float)(unsafe.Pointer(&b[0])), C.__LAPACK_int(p),
		0, (*C.float)(unsafe.Pointer(&out[0])), C.__LAPACK_int(p))
}
```

Функцию достаточно подключить в `tensor.MatMul`. Файл стоит пометить `//go:build darwin`, чтобы на других ОС остался чистый Go. Для matmul в backward нужны варианты с транспонированием: `CblasTrans` вместо `CblasNoTrans`, без лишних копий.

## 2. Metal: настоящий GPU на Mac

Есть три пути:

- **MPS**: `MPSMatrixMultiplication` вызывается через cgo и тонкую обёртку на Objective-C.
- **Свои compute shaders на MSL** для softmax, layernorm, GELU и backward.
- **mlx-c**: C API от Apple MLX, тоже через cgo. Там уже есть готовые ядра.

Плюс Mac в общей памяти CPU и GPU (unified memory). `newBufferWithBytesNoCopy` оборачивает Go-слайс без копирования, но память должна быть выровнена по странице.

## 3. CUDA, если будет NVIDIA (например, арендованная машина)

cgo плюс cuBLAS/cuDNN. Готовые биндинги вроде `gorgonia.org/cu` давно заброшены, проще написать свои.

## 4. Кроссплатформенно

WebGPU (wgpu-native, Go-биндинги `cogentcore/webgpu`) или Vulkan compute. Один код будет работать и на Mac, и на NVIDIA, но порог входа выше.

## Главная ловушка

Если отправлять на GPU только matmul, а всё остальное считать на CPU, ускорения почти не будет. Время уйдёт на синхронизацию и запуск ядер, особенно на маленькой модели. Реальный выигрыш появится, когда тензоры живут на устройстве (device-resident) и все операции forward и backward сделаны как ядра. Для учебного GPT это фактически переписывание `tensor` и `autograd` под отдельный backend.

## Рекомендация

Начать с Accelerate: это один файл и примерно 15× на matmul. Metal брать, если хочется разобраться, как устроены GPU-ядра.
