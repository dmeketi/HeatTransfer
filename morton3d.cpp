#include <algorithm>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdint.h>
#include <chrono>
inline int idx(int k, int i, int j, int M) {
    return k*M*M + i*M + j;
}

void print_slice(const std::vector<double>& X, int M, int k, const char* name) {
    std::cout << name << " (k=" << k << ")\n";
    std::cout << std::fixed << std::setprecision(0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            std::cout << std::setw(3) << X[k*M*M + i*M + j];
        }
        std::cout << '\n';
    }
    std::cout << '\n';
} //this is horrible for large M, find 3D solution like paraview

void save_vtk(const std::vector<double>& A, int M, const std::string& filename) {
    std::ofstream out("timeFiles\\" + filename);
    out << "# vtk DataFile Version 3.0\n";
    out << "Heat diffusion\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << M << " " << M << " " << M << "\n";
    out << "ORIGIN 0 0 0\n";
    out << "SPACING 1 1 1\n";
    out << "POINT_DATA " << M*M*M << "\n";
    out << "SCALARS temperature float 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int k = 0; k < M; ++k)
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < M; ++j)
                out << A[idx(k,i,j,M)] << "\n";
}

inline uint32_t bitSpreader(uint32_t x) {
        //we wont need more than 10 bits so mask the rest to ensure 0'd
        x &= 0x000003ff; //0000 0000 0000 0011 1111 1111

        //these 'mask' numbers will start spreading the 1's out and put two 0's in between them
        x = (x | (x << 16)) & 0x030000FF;   //part 1 0000 0011 0000 0000 0000 0000 1111 1111
        x = (x | (x <<  8)) & 0x0300F00F;   //2
        x = (x | (x <<  4)) & 0x030C30C3;   //3
        x = (x | (x <<  2)) & 0x09249249;   //final pattern should have bits at places 0,3,6, (two 0's between)

        return x;
}

inline uint32_t mortonKeyMaker(uint32_t bk, uint32_t bi, uint32_t bj) {
    uint32_t K = bitSpreader(bk) << 2;  // lane 2
    uint32_t I = bitSpreader(bi) << 1;  // lane 1
    uint32_t J = bitSpreader(bj);       // lane 0
    return K | I | J;
}

struct Tile { uint32_t bk, bi, bj, key; };

inline std::vector<Tile> makeMortonTiles(int M, int TILE) {
    const int Nt = M / TILE;
    std::vector<Tile> tiles;
    for (uint32_t bk = 0; bk < (uint32_t)Nt; ++bk)
        for (uint32_t bi = 0; bi < (uint32_t)Nt; ++bi)
            for (uint32_t bj = 0; bj < (uint32_t)Nt; ++bj)
                tiles.push_back({bk, bi, bj, mortonKeyMaker(bk, bi, bj)});
    std::sort(tiles.begin(), tiles.end(),
              [](const Tile& a, const Tile& b){ return a.key < b.key; });

    if (M % TILE != 0) { std::cerr << "M%TILE != 0\n"; std::exit(1); }
    return tiles;
}

int main() {
    const int M = 64; // size of the cube
    const int STEPS = 50; // how many frames (files)
    const int TILE = 16;

    const auto tiles = makeMortonTiles(M, TILE);

    std::vector<double> A(M * M * M); //before
    std::vector<double> B(M * M * M); //after
    std::fill(A.begin(), A.end(), 0.0);
    std::fill(B.begin(), B.end(), 0.0); //initalizing everything to 0
    //we'll make a hot spot in the middle of A to see how the heat spreads in B
    int mid = M / 2;
    A[idx(mid, mid, mid, M)] = 9999999999;

    save_vtk(A, M, "heat_000.vtk"); //in the beginning...
    const auto start = std::chrono::high_resolution_clock::now();

    for (int t = 1; t <= STEPS; ++t) {
        for (int k = 1; k < M-1; ++k) {
            //depth (z)
            for (int i = 1; i < M-1; ++i) {
                //row (y)
                for (int j = 1; j < M-1; ++j) {
                    //column (x)
                    int center  = idx(k, i, j, M);
                    //we need to find the 6 neighbor indices, legend = xm - > "x [axis] minus"
                    int xm = idx(k,i,j-1,M);
                    int xp = idx(k,i,j+1,M);
                    int ym = idx(k,i-1,j,M);
                    int yp = idx(k,i+1,j,M);
                    int zm = idx(k-1,i,j,M);
                    int zp = idx(k+1,i,j,M);
                    B[center] = (A[center] + A[xm] + A[xp] + A[ym] + A[yp] + A[zm] + A[zp]) / 7.0;
                }
            }
        }
        std::swap(A, B); //swapping to the new block after computing
        std::ostringstream name;
        name << "heat_" << std::setfill('0') << std::setw(3) << t << ".vtk";
        save_vtk(A, M, name.str()); //now we put this layer of B in the file

    }
    const auto end = std::chrono::high_resolution_clock::now(); //measure computation

    auto result = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Computation took %lld nanoseconds", result);

    return 0;
}