#include <iostream>
#include <omp.h>

int main() {
    const int size = 1000;
    int sum = 0;

    // Arreglo de prueba
    int array[size];
    for (int i = 0; i < size; i++) {
        array[i] = i + 1;  // Llena el arreglo con los números del 1 al 1000
    }

    // Paraleliza la suma de los elementos del arreglo
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }

    std::cout << "La suma de los elementos del arreglo es: " << sum << std::endl;
    return 0;
}
