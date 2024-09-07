#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <math.h>
#include <time.h>

#define LEARN_RATE 1e-6
#define TRAINING_DATA_SIZE (sizeof(training_data) / sizeof(training_data[0]))
#define TEST_DATA_SIZE (sizeof(test_data) / sizeof(test_data[0]))

typedef struct {
  double w1;
  double w2;
} Model;

double training_data[][3] = {
  {1, 2, 3}, {2, 4, 7}, {0, 0, 0}, {0, 9, 9}, {1, 9, 10}, {2, 3, 5},
  {1, 1, 2}, {6, 2, 8}, {5, 5, 10}, {4, 6, 10}, {3, 4, 7}, {2, 5, 7},
  {0, 5, 5}, {1, 4, 5}, {7, 3, 10}, {9, 1, 10}, {0, 10, 10}, {2, 2, 4},
  {3, 6, 9}, {4, 1, 5}, {6, 4, 10}, {8, 2, 10}, {1, 5, 6}, {7, 2, 9},
  {3, 3, 6},
};

double test_data[][3] = {
  {0, 1, 1}, {2, 6, 8}, {4, 4, 8}, {5, 3, 8}, {3, 5, 8}, {2, 8, 10},
  {6, 3, 9}, {1, 8, 9}, {5, 2, 7}, {7, 1, 8}, {9, 0, 9}, {3, 7, 10},
  {8, 1, 9}, {4, 5, 9}, {2, 7, 9},
};

Model model;

double get_rand_d(double a) {
  return a * ((double)rand() / RAND_MAX);
}

void handle_int(int sig) {
  for (int i = 0; i < TEST_DATA_SIZE; i++) {
    double x1 = test_data[i][0];
    double x2 = test_data[i][1];
    double y = x1 * model.w1 + x2 * model.w2;
    printf("Predicted: %f + %f = %f (Rounded: %f)\n", x1, x2, y, round(y));
  }
  exit(1);
}

double compute_cost(Model model) {
  double acc = 0;

  for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
    double x1 = training_data[i][0];
    double x2 = training_data[i][1];
    double y = training_data[i][2];

    double model_y = x1 * model.w1 + x2 * model.w2;
    double mse = (model_y - y);
    acc += mse * mse;
  }

  return acc / TRAINING_DATA_SIZE;
}

Model compute_cost_derivative(Model model) {
  Model derivative = {0, 0};

  for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
    double x1 = training_data[i][0];
    double x2 = training_data[i][1];
    double y = training_data[i][2];

    double model_y = x1 * model.w1 + x2 * model.w2;
    double error = model_y - y;

    derivative.w1 += error * x1;
    derivative.w2 += error * x2;
  }

  derivative.w1 *= 2.0 / TRAINING_DATA_SIZE;
  derivative.w2 *= 2.0 / TRAINING_DATA_SIZE;

  return derivative;
}

int main() {
  srand(time(NULL));
  model = (Model){get_rand_d(5), get_rand_d(5)};
  signal(SIGINT, handle_int);

  for (;;) {
    Model derivative = compute_cost_derivative(model);
    model.w1 -= derivative.w1 * LEARN_RATE;
    model.w2 -= derivative.w2 * LEARN_RATE;
    printf("Model: w1=%f, w2=%f, Cost=%f\n", model.w1, model.w2, compute_cost(model));
  }

  handle_int(0);
  return 0;
}
