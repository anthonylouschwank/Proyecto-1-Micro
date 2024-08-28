#include <vector>
#include <omp.h>
#include <numeric>
#include <cmath>

// Función C++ para calcular la elongación sin OpenCV
extern "C" void calculateElongation(const double* mu20, const double* mu02, const double* mu11, int size, double* elongations) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        double x = mu20[i] + mu02[i];
        double y = 4 * mu11[i] * mu11[i] + (mu20[i] - mu02[i]) * (mu20[i] - mu02[i]);
        elongations[i] = (x + std::sqrt(y)) / (x - std::sqrt(y));
    }
}
