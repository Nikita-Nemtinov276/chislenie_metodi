```
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <algorithm>

using namespace std;

// ФОРМАТИРОВАНИЕ ЧИСЕЛ
string formatNumber(double value, int precision = 6) {
    if (abs(value) < 1e-15) return "0";

    stringstream ss;
    double av = abs(value);

    if (av >= 1e-3 && av < 1e4) {
        ss << fixed << setprecision(precision) << value;
        string str = ss.str();

        // Удаляем лишние нули после запятой
        size_t dotPos = str.find('.');
        if (dotPos != string::npos) {
            while (!str.empty() && str.back() == '0')
                str.pop_back();
            if (!str.empty() && str.back() == '.')
                str.push_back('0');
        }
        return str;
    }
    else {
        ss << scientific << setprecision(precision) << value;
        return ss.str();
    }
}

// ГАУСС
vector<double> gaussSolve(const vector<vector<double>>& A,
    const vector<double>& b)
{
    int n = static_cast<int>(A.size());
    if (n == 0) throw invalid_argument("Пустая матрица A");

    vector<vector<double>> M(n, vector<double>(n + 1));

    // Формирование расширенной матрицы [A | b]
    for (int i = 0; i < n; i++) {
        if (A[i].size() != static_cast<size_t>(n))
            throw invalid_argument("Матрица A не квадратная");
        for (int j = 0; j < n; j++)
            M[i][j] = A[i][j];
        M[i][n] = b[i];
    }

    // Прямой ход
    for (int k = 0; k < n; k++) {

        // Выбор ведущего элемента по модулю в столбце k
        int pivotRow = k;
        double maxVal = abs(M[k][k]);

        for (int i = k + 1; i < n; i++) {
            if (abs(M[i][k]) > maxVal) {
                maxVal = abs(M[i][k]);
                pivotRow = i;
            }
        }

        if (maxVal < numeric_limits<double>::epsilon())
            throw runtime_error("Матрица вырождена (нулевой столбец).");

        if (pivotRow != k)
            swap(M[k], M[pivotRow]);

        double pivot = M[k][k];

        // Нормализация ведущей строки
        for (int j = k; j <= n; j++)
            M[k][j] /= pivot;

        // Обнуление элементов ниже
        for (int i = k + 1; i < n; i++) {
            double factor = M[i][k];
            if (abs(factor) < 1e-15) continue;
            for (int j = k; j <= n; j++)
                M[i][j] -= factor * M[k][j];
        }
    }

    // Обратный ход
    vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = M[i][n];
        for (int j = i + 1; j < n; j++)
            x[i] -= M[i][j] * x[j];
    }

    return x;
}

// Проверка решения
void CheckSolution(const vector<vector<double>>& A,
    const vector<double>& x,
    const vector<double>& b)
{
    cout << "\nПроверка решения (подстановка в уравнения):\n";

    int n = A.size();
    cout << fixed << setprecision(10);

    for (int i = 0; i < n; i++) {
        cout << "\nУравнение " << i + 1 << ":\n";

        // Печать уравнения с коэффициентами
        for (int j = 0; j < n; j++) {
            cout << "  (" << formatNumber(A[i][j]) << ")*x" << (j + 1);
            if (j + 1 < n) cout << " +";
            cout << "\n";
        }

        // Печать подстановки x
        cout << "\nПодстановка x:\n";
        for (int j = 0; j < n; j++) {
            cout << "  (" << formatNumber(A[i][j]) << ")*(" << formatNumber(x[j]) << ")";
            if (j + 1 < n) cout << " +";
            cout << "\n";
        }

        // Вычисление результата
        double sum = 0.0;
        for (int j = 0; j < n; j++)
            sum += A[i][j] * x[j];

        cout << "\nРезультат: " << sum << "\nОжидалось: " << b[i] << "\n";
        cout << "Разница: " << (sum - b[i]) << "\n";
    }
}


// КЛАСС SOR (РЕЛАКСАЦИИ)
class RelaxationSolver {
private:
    int n;
    double omega;
    double epsilon;
    int maxSteps;

    vector<vector<double>> A;
    vector<double> b;
    vector<double> x;

public:
    RelaxationSolver(const vector<vector<double>>& A_,
        const vector<double>& b_,
        double eps = 1e-6,
        double w = 1.0,
        int maxIter = 1000)
        : A(A_), b(b_), epsilon(eps), omega(w), maxSteps(maxIter)
    {
        n = static_cast<int>(A.size());
        if (n == 0) throw invalid_argument("Пустая матрица A");
        if (b.size() != static_cast<size_t>(n))
            throw invalid_argument("Размер b не совпадает с размером A");

        for (int i = 0; i < n; ++i) {
            if (A[i].size() != static_cast<size_t>(n))
                throw invalid_argument("Матрица A не квадратная");
            if (abs(A[i][i]) < 1e-15)
                throw runtime_error("Нулевой диагональный элемент в строке " + to_string(i + 1));
        }

        x.assign(n, 0.0);
    }

    void setInitialGuess(const vector<double>& guess) {
        if (guess.size() != static_cast<size_t>(n))
            throw invalid_argument("Неверный размер начального приближения");
        x = guess;
    }

    // Норма невязки ||Ax - b||
    double residualNorm() const {
        double rmax = 0.0;
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++)
                sum += A[i][j] * x[j];
            rmax = max(rmax, abs(sum - b[i]));
        }
        return rmax;
    }

    // Проверка диагонального преобладания
    bool checkDiagonalDominance() const {
        bool strict = true;
        cout << "Проверка диагонального преобладания:\n";
        for (int i = 0; i < n; i++) {
            double diag = abs(A[i][i]);
            double sum = 0.0;
            for (int j = 0; j < n; j++)
                if (i != j) sum += abs(A[i][j]);

            cout << "  Строка " << i + 1 << ": |" << formatNumber(A[i][i])
                << "| " << (diag > sum ? ">" : "<=") << " "
                << formatNumber(sum) << "\n";

            if (diag <= sum) strict = false;
        }
        if (!strict)
            cout << "ВНИМАНИЕ: строгого диагонального преобладания нет, сходимость не гарантируется.\n";
        cout << "\n";
        return strict;
    }

    // Метод релаксации SOR
    bool solve() {
        cout << "\nМЕТОД РЕЛАКСАЦИИ SOR\n";
        cout << "omega = " << omega << ", epsilon = " << epsilon << ", maxSteps = " << maxSteps << "\n\n";

        checkDiagonalDominance();

        vector<double> x_new(n);
        int step = 0;

        cout << setw(8) << "Итерация"
            << setw(15) << "Скачок"
            << setw(15) << "Невязка"
            << setw(15) << "x1"
            << setw(15) << "x2"
            << setw(15) << "x3" << "\n";

        cout << string(83, '-') << "\n";

        for (; step < maxSteps; step++) {

            double dx_max = 0.0;

            for (int i = 0; i < n; i++) {
                double sigma = 0.0;

                // Используем уже обновлённые значения x[j] для j < i
                for (int j = 0; j < i; j++)
                    sigma += A[i][j] * x_new[j];

                // Используем старые значения x[j] для j > i
                for (int j = i + 1; j < n; j++)
                    sigma += A[i][j] * x[j];

                double xi_old = x[i];

                // Формула SOR:
                // x_i^{k+1} = x_i^k + omega * ( (b_i - sigma) / a_ii - x_i^k )
                x_new[i] = x[i] + omega * ((b[i] - sigma) / A[i][i] - x[i]);

                dx_max = max(dx_max, abs(x_new[i] - xi_old));
            }

            x = x_new;

            double r = residualNorm();

            if (step < 10 || step % 10 == 0 || r < epsilon) {
                cout << setw(8) << step
                    << setw(15) << formatNumber(dx_max)
                    << setw(15) << formatNumber(r);

                for (int i = 0; i < min(n, 3); i++)
                    cout << setw(15) << formatNumber(x[i]);

                cout << "\n";
            }

            if (dx_max < epsilon && r < epsilon)
                break;
        }

        if (step >= maxSteps) {
            cout << "\nДостигнут лимит итераций, решение может быть неточным.\n";
            return false;
        }

        cout << "\nСходимость достигнута за " << step << " итераций.\n";
        return true;
    }

    const vector<double>& getSolution() const { return x; }

    void printSolution() const {
        cout << "\nРЕШЕНИЕ (SOR):\n";
        for (int i = 0; i < n; i++)
            cout << "  x" << i + 1 << " = " << formatNumber(x[i], 10) << "\n";
    }
};

int main() {
    // Система
    vector<vector<double>> A = {
        {0.78, -0.02, -0.12, -0.14},
        {-0.02, 0.86, -0.04, -0.06},
        {-0.12, -0.04, 0.72, -0.08},
        {-0.14,  0.06,  0.08, 0.74}
    };

    vector<double> b = { 0.76, 0.08, 1.12, 0.74 };

    double epsilon, omega;
    int maxSteps;

    cout << "Введите epsilon (по умолчанию 1e-6): ";
    string s;
    getline(cin, s);
    epsilon = s.empty() ? 1e-6 : stod(s);

    cout << "Введите omega (0 < omega < 2, по умолчанию 1.0): ";
    getline(cin, s);
    omega = s.empty() ? 1.0 : stod(s);

    cout << "Введите maxSteps (по умолчанию 1000): ";
    getline(cin, s);
    maxSteps = s.empty() ? 1000 : stoi(s);

    try {
        // Метод Гаусса
        vector<double> gaussSol = gaussSolve(A, b);

        cout << "\nРЕШЕНИЕ (ГАУСС):\n";
        for (int i = 0; i < static_cast<int>(gaussSol.size()); i++)
            cout << "  x" << i + 1 << " = "
            << formatNumber(gaussSol[i], 10) << "\n";

        CheckSolution(A, gaussSol, b);

        // ---------- Метод релаксации (SOR) ----------
        RelaxationSolver solver(A, b, epsilon, omega, maxSteps);
        solver.setInitialGuess({ 0.0, 0.0, 0.0, 0.0 });

        if (solver.solve())
            solver.printSolution();
    }
    catch (const exception& e) {
        cout << "Ошибка: " << e.what() << "\n";
    }

    return 0;
}

```
