# ifndef TESTS_H
# define TESTS_H

# include <vector>

using namespace std;

void writeToFile(const char* folder, char *fileName, int sm, double value);
void writeToFile(const char* folder, char *fileName, int sm, vector<double> values);
void writeToFile(const char *fileName, int sm, vector<double> values);

void testSpeedup(char **argv);
// void testConcurrency(char **argv);
void testTailing(char **argv);

# endif