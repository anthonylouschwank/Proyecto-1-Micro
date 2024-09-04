#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>
#include <sstream>

// Función C++ para calcular la elongación sin OpenCV
extern "C" void calculateElongation(const double *mu20, const double *mu02,
                                    const double *mu11, int size,
                                    double *elongations) {
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    double x = mu20[i] + mu02[i];
    double y =
        4 * mu11[i] * mu11[i] + (mu20[i] - mu02[i]) * (mu20[i] - mu02[i]);
    elongations[i] = (x + std::sqrt(y)) / (x - std::sqrt(y));
  }
}

// Main function to receive arguments from Python
int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: ./elongation <mu20_values> <mu02_values> <mu11_values>"
              << std::endl;
    return 1;
  }

  // Convertir los argumentos en arrays de double
  std::vector<double> mu20, mu02, mu11;
  std::istringstream mu20_stream(argv[1]);
  std::istringstream mu02_stream(argv[2]);
  std::istringstream mu11_stream(argv[3]);

  double value;
  while (mu20_stream >> value) mu20.push_back(value);
  while (mu02_stream >> value) mu02.push_back(value);
  while (mu11_stream >> value) mu11.push_back(value);

  int size = mu20.size();
  std::vector<double> elongations(size);

  // Llamar a la función para calcular la elongación
  calculateElongation(mu20.data(), mu02.data(), mu11.data(), size, elongations.data());

  // Imprimir los resultados (elongaciones) como salida estándar
  for (const auto &elongation : elongations) {
    std::cout << elongation << " ";
  }

  std::cout << std::endl;

  return 0;
}

