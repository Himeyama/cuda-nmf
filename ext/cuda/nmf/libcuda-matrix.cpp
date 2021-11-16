#include <iostream>
#include <cublas.h>
#include <random>
#include <typeinfo>
#include <list>

template <class T>
class CuMatrix{  
    public:
    T *dMat;
    std::int64_t rowSize;
    std::int64_t colSize;
    bool alloced;

    CuMatrix(){};

    CuMatrix(std::int64_t row, std::int64_t col, T *mat = NULL, bool mode = true){
        // mode == true ? host : device
        rowSize = row;
        colSize = col;
        alloced = false;
        if(mode){
            cublasAlloc(row * col, sizeof(T), (void**)&dMat);
            alloced = true;
        }else{
            dMat = mat;
            alloced = true;
        }
        
        bool zero = false;
        if(mat == NULL){
            mat = (T*)malloc(sizeof(T) * row * col);
            memset(mat, 0, sizeof(T) * row * col);
            zero = true;
        }
        cublasSetMatrix(row, col, sizeof(T), mat, rowSize, dMat, rowSize);
        if(zero) free(mat);
    }

    static CuMatrix rand(std::int64_t row, std::int64_t col){
        std::random_device rd;
        std::default_random_engine engine(rd());
        std::uniform_real_distribution<> urd(0, 1);
        T *x = (T*)malloc(sizeof(T) * row * col);
        for(std::int64_t i = 0; i < row * col; i++)
            x[i] = urd(engine);
        CuMatrix<T> r(row, col, x);
        free(x);
        return r;
    }

    CuMatrix copy(){
        CuMatrix<T> cp(rowSize, colSize);
        cublasCopy(rowSize * colSize, dMat, 1, cp.dMat, 1);
        return cp;
    }

    static CuMatrix I(std::int64_t n, T k = 1){
        T *x = (T*)malloc(sizeof(T) * n * n);
        memset(x, 0, sizeof(T) * n * n);
        for(std::int64_t i = 0; i < n; i++)
            x[i * n + i] = k;
        CuMatrix<T> r(n, n, x);
        return r;
    }

    void freeMat(){
        if(alloced){
            cublasFree(dMat);
            dMat = NULL;
        }
        alloced = false;
    }

    // 型によって関数を分ける
    void cublasGemm(char transa, char transb, std::int64_t m, std::int64_t n, std::int64_t k, float alpha, const float *A, std::int64_t lda, const float *B, std::int64_t ldb, float beta, float *C, std::int64_t ldc){
        cublasSgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    void cublasGemm(char transa, char transb, std::int64_t m, std::int64_t n, std::int64_t k, double alpha, const double *A, std::int64_t lda, const double *B, std::int64_t ldb, double beta, double *C, std::int64_t ldc){
        cublasDgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); 
    }

    // OK
    CuMatrix<T> operator * (CuMatrix<T> b){
        if (colSize != b.rowSize)
            std::cerr << "*(CuMatrix): 行列1の行数と行列2の列数が異なります" << std::endl;
        // std::int64_t m = rowSize, n = b.colSize, k = colSize;
        std::int64_t m = rowSize, n = b.colSize, k = colSize;
        CuMatrix<T> c(m, n);
        cublasGemm('N', 'N', n, m, k, 1, b.dMat, n, dMat, k, 0, c.dMat, n);
        return c;
    }

    // OK
    void dot(CuMatrix<T> b, T* ptr = NULL){
        if(colSize != b.rowSize)
            std::cerr << "dot(): 行列1の行数と行列2の列数が異なります" << std::endl;
        std::int64_t m = rowSize, n = b.colSize, k = colSize;
        cublasGemm('N', 'N', n, m, k, 1, b.dMat, n, dMat, k, 0, ptr, n);
    }

    // OK
    void tdot(CuMatrix<T> b, T* ptr = NULL){
        if(rowSize != b.rowSize)
            std::cerr << "tdot(): 行列1の列数と行列2の列数が異なります" << std::endl;
        std::int64_t m = colSize, n = b.colSize, k = rowSize;
        cublasGemm('N', 'T', n, m, k, 1, b.dMat, n, dMat, m, 0, ptr, n);
    }

    // OK
    void dott(CuMatrix<T> b, T* ptr = NULL){
        if(colSize != b.colSize)
            std::cerr << "dott(): 行列1の列数と行列2の列数が異なります" << std::endl;
        std::int64_t m = rowSize, n = b.rowSize, k = colSize;
        cublasGemm('T', 'N', n, m, k, 1, b.dMat, k, dMat, k, 0, ptr, n);
    }

    void cublasAxpy(std::int64_t n, float alpha, const float *x, std::int64_t incx, float *y, std::int64_t incy){
        cublasSaxpy(n, alpha, x, incx, y, incy);
    }

    void cublasAxpy(std::int64_t n, double alpha, const double *x, std::int64_t incx, double *y, std::int64_t incy){
        cublasDaxpy(n, alpha, x, incx, y, incy);
    }

    void operator +=(CuMatrix<T> b){
        if(rowSize == 1 || colSize == 1 || rowSize * colSize == b.rowSize * b.colSize)
            ;// ベクトルの足し算
        else if(rowSize != b.rowSize || colSize != b.colSize)
            std::cerr << "+(CuMatrix): 二つの行列の大きさが異なります" << std::endl;
        cublasAxpy(rowSize * colSize, 1, b.dMat, 1, dMat, 1);
    }

    void operator -=(CuMatrix<T> b){
        if(rowSize == 1 || colSize == 1 || rowSize * colSize == b.rowSize * b.colSize)
            ;// ベクトルの引き算
        else if(rowSize != b.rowSize || colSize != b.colSize)
            std::cerr << "-(CuMatrix): 二つの行列の大きさが異なります" << std::endl;
        cublasAxpy(rowSize * colSize, -1, b.dMat, 1, dMat, 1);
    }

    void operator *=(CuMatrix<T> b){
        cublasGemm('N', 'N', rowSize, colSize, colSize, 1, dMat, rowSize, b.dMat, b.rowSize, 1, dMat, rowSize);
    }

    CuMatrix<T> operator *(T b){
        CuMatrix<T> matB = I(colSize, b);
        return *this *(matB);
    }

    CuMatrix<T> times(CuMatrix<T> b){
        CuMatrix<T> r;
        if(!(b.colSize == colSize && b.rowSize == rowSize)){
            std::cerr << "times(): 二つの行列の大きさが異なります" << std::endl;
            return r;
        }
        T *h_a = toMem();
        T *h_b = b.toMem();
        T *h_c = (T*)malloc(sizeof(T) * rowSize * colSize);
        for(std::int64_t i = 0; i < rowSize * colSize; i++)
            h_c[i] = h_a[i] * h_b[i];
        r = CuMatrix(rowSize, colSize, h_c);
        free(h_a);
        free(h_b);
        free(h_c);
        return r;
    }

    CuMatrix<T> rdivide(CuMatrix<T> b){
        CuMatrix<T> r;
        if(!(b.colSize == colSize && b.rowSize == rowSize)){
            std::cerr << "rdivide(): 二つの行列の大きさが異なります" << std::endl;
            return r;
        }
        T *h_a = toMem();
        T *h_b = b.toMem();
        T *h_c = (T*)malloc(sizeof(T) * rowSize * colSize);
        for(std::int64_t i = 0; i < rowSize * colSize; i++){
            if(h_b[i] == 0)
                std::cerr << "rdivide(): ゼロ除算" << std::endl;
            h_c[i] = h_a[i] / h_b[i];
        }
        r = CuMatrix(rowSize, colSize, h_c);
        free(h_a);
        free(h_b);
        free(h_c);
        return r;
    }

    void inspect(){
        T *mat = toMem();
        std::string str = "[";
        for(std::int64_t i = 0; i < rowSize; i++){
            str += "[";
            for(std::int64_t j = 0; j < colSize; j++){
                str += std::to_string(mat[i * colSize + j]) + (j == colSize - 1 ? "" : ", ");
            }
            str += (i == rowSize - 1 ? "]" : "], ");
        }
        str += "] (" +  std::to_string(rowSize) + ", " + std::to_string(colSize) + ")";
        std::cout << str << std::endl;
        free(mat);
    }

    T* toMem(){
        T *mat = (T*)malloc(sizeof(T) * rowSize * colSize);
        cublasGetMatrix(rowSize, colSize, sizeof(T), dMat, rowSize, mat, rowSize);
        return mat;
    }

    void cublasCopy(std::int64_t n, const float *x, std::int64_t incx, float *y, std::int64_t incy){
        cublasScopy(n, x, incx, y, incy);
    }

    void cublasCopy(std::int64_t n, const double *x, std::int64_t incx, double *y, std::int64_t incy){
        cublasDcopy(n, x, incx, y, incy);
    }

    void getRow(std::int64_t i, T* d_vec = NULL){
        if(d_vec == NULL)
            cublasAlloc(colSize, sizeof(T), (void**)&d_vec);
        cublasCopy(colSize, dMat + colSize * i, 1, d_vec, 1);
    }

    void getCol(std::int64_t i, T* d_vec = NULL){
        if(d_vec == NULL)
            cublasAlloc(rowSize, sizeof(T), (void**)&d_vec);
        cublasCopy(rowSize, dMat + i, colSize, d_vec, 1);
    }

    void setRow(std::int64_t i, CuMatrix<T> b){
        cublasCopy(colSize, b.dMat, 1, dMat + colSize * i, 1);
    }

    void setCol(std::int64_t i, CuMatrix<T> b){
        cublasCopy(rowSize, b.dMat, 1, dMat + i, colSize);
    }

    // 型によって分ける
    float cublasDot(int n, const float *x, int incx, const float *y, int incy){
        return cublasSdot(n, x, incx, y, incy);
    }
    double cublasDot(int n, const double *x, int incx, const double *y, int incy){
        return cublasDdot(n, x, incx, y, incy);
    }

    // 行列の2乗和
    T sumSq(){
        return cublasDot(rowSize * colSize, dMat, 1, dMat, 1);
    }
};

template class CuMatrix<float>;
template class CuMatrix<double>;
