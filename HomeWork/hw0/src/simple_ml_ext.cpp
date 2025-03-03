#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int itera_num = (int)(m/batch) 
    for(int i = 0;i < itera_num;i++){
        softmax_regression_batch_cpp(X+batch*n*i,y+batch*n*i,theta,m,n,k,lr,batch)
    }
    if(itera_num*batch<m){
        softmax_regression_epoch_cpp(X+batch*n*itera_num,y+batch*n*itera_num,theta,m,n,k,lr,m-itera_num*batch)
    }
    /// END YOUR CODE
}
void matmul(const float* X,float* Y,float* O,size_t m,size_t n,size_t k)
{
    
    for(int i = 0;i<m;i++){
        for(int j =0;j<m;j++){
            for(int f=0;f<k;f++){
                O[m][n] += X[m][k]*Y[k][n]
            }
        }
    }
}
void matmul(float* X,float* Y,float* O,size_t m,size_t n,size_t k)
{
    
    for(int i = 0;i<m;i++){
        for(int j =0;j<m;j++){
            for(int f=0;f<k;f++){
                O[m][n] += X[m][k]*Y[k][n]
            }
        }
    }
}
void softmax_regression_batch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t n, size_t k,
								  float lr,size_t batch)
{
    int m = batch;
    t = (float*)malloc(sizeof(float)*m*k)

    matmul(X,theta,t,m,k,n)
    for(int i = 0;i<m;i++){
        for(int j = 0;j<k;j++){
            t[i][j] = math.exp(t[i][j])
        }
    }

    for(int i = 0;i<m;i++){
        float sum = 0;
        for(int j = 0;j<k;j++){
            sum += t[i][j]
        }
        
        for(int j = 0;j<k;j++){
            t[i][j] /= sum;
        }
    }

    
    one_hot = (float*)malloc(sizeof(float)*m*k)
    memset(one_hot,0,sizeof(float)*m*k)
    for(int i =0;i<m;i++){
        one_hot[i][y[i]-'0'] =1;
    }

    for(int i = 0;i<m;i++){
        for(int j = 0;j<k;j++){
            t[i][j] -= one_hot[i][j]

        }
    }
    
    deta = (float*)malloc(sizeof(float)*n*k)
    XT = (float*)malloc(sizeof(float)*m*n)
    for(int i = 0;i< n;i++){
        for(int j = 0;j<m;j++){
            XT[i][j] = X[j][i]
        }
    }
    matmul(XT,t,deta,n,k,m)
    
    for(int i = 0;i< n;i++){
        for(int j = 0;j<m;j++){
            XT[i][j] /= m;
        }
    }
    
    for(int i = 0;i< n;i++){
        for(int j = 0;j<k;j++){
            theta[i][j] -= lr*deta[i][j]
        }
    }
    free(t)
    free(one_hot)
    free(deta)
    free(XT)
}
/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
