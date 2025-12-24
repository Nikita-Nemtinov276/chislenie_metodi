Задача написать код для решения матрицы Методом Гаусса и Методом релаксации.

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

// ---- Функция Гаусса с обработкой бесконечно многих решений ----
vector<double> gaussSolve(const vector<vector<double>>& A,
    const vector<double>& b) {
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

    int rankA = 0; // ранг матрицы A
    int rankAug = 0; // ранг расширенной матрицы

    // Прямой ход с частичным выбором ведущего элемента
    for (int k = 0; k < n; k++) {
        // Поиск строки с максимальным элементом в столбце k
        int pivotRow = -1;
        double maxVal = 0.0;
        for (int i = rankA; i < n; i++) {
            if (abs(M[i][k]) > maxVal) {
                maxVal = abs(M[i][k]);
                pivotRow = i;
            }
        }

        if (maxVal < 1e-15) continue; // свободная переменная

        swap(M[pivotRow], M[rankA]);

        double pivot = M[rankA][k];
        for (int j = k; j <= n; j++)
            M[rankA][j] /= pivot;

        for (int i = 0; i < n; i++) {
            if (i == rankA) continue;
            double factor = M[i][k];
            if (abs(factor) < 1e-15) continue;
            for (int j = k; j <= n; j++)
                M[i][j] -= factor * M[rankA][j];
        }
        
        rankA++;
    }

    cout << "\nКонечная ступенчатая матрица [A|b]:\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= n; j++)
            cout << setw(12) << formatNumber(M[i][j]) << " ";
        cout << "\n";
    }

    // Вычисляем ранг расширенной матрицы
    rankAug = 0;
    for (int i = 0; i < n; i++) {
        bool nonZero = false;
        for (int j = 0; j <= n; j++) {
            if (abs(M[i][j]) > 1e-15) {
                nonZero = true;
                break;
            }
        }
        if (nonZero) rankAug++;
    }

    if (rankA < rankAug) {
        throw runtime_error("Система несовместна → 0 решений");
    }
    else if (rankA < n) {
        cout << "Решений бесконечно много\n";
    }

    // Обратный ход (свободные переменные = 0)
    vector<double> x(n, 0.0);

    for (int i = n - 1; i >= 0; i--) {
        int pivotCol = -1;
        for (int j = 0; j < n; j++) {
            if (abs(M[i][j]) > 1e-15) {
                pivotCol = j;
                break;
            }
        }
        if (pivotCol == -1) continue; // строка из нулей
        double sum = 0.0;
        for (int j = pivotCol + 1; j < n; j++)
            sum += M[i][j] * x[j];
        x[pivotCol] = M[i][n] - sum;
    }

    return x;
}


// Проверка решения
void CheckSolution(const vector<vector<double>>& A,
    const vector<double>& x,
    const vector<double>& b) {
    cout << "\nПроверка решения (подстановка в уравнения):\n";

    int n = A.size();
    cout << fixed << setprecision(10);

    for (int i = 0; i < n; i++) {
        cout << "\nУравнение " << i + 1 << ":\n";

        for (int j = 0; j < n; j++) {
            cout << "  (" << formatNumber(A[i][j]) << ")*x" << (j + 1);
            if (j + 1 < n) cout << " +";
            cout << "\n";
        }

        cout << "\nПодстановка x:\n";
        for (int j = 0; j < n; j++) {
            cout << "  (" << formatNumber(A[i][j]) << ")*(" << formatNumber(x[j]) << ")";
            if (j + 1 < n) cout << " +";
            cout << "\n";
        }

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
        : A(A_), b(b_), epsilon(eps), omega(w), maxSteps(maxIter) {
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

                for (int j = 0; j < i; j++)
                    sigma += A[i][j] * x_new[j];

                for (int j = i + 1; j < n; j++)
                    sigma += A[i][j] * x[j];

                double xi_old = x[i];

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
    /*vector<vector<double>> A = {
    {1, 1},
    {1, 1}
    };

    vector<double> b = { 2, 3 };*/



    vector<vector<double>> A = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9}
    };

    vector<double> b = { 3, 6, 9 };

    /*vector<vector<double>> A = {
    {1, 1},
    {2, 2}
    };

    vector<double> b = { 2, 4 };*/


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
