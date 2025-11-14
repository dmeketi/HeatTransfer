#include <algorithm>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <Kokkos_Core.hpp>
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
} //this is impossible to look at for large M, find 3D solution like paraview

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


struct Tile { uint32_t bk, bi, bj, key; };



int main() {
    const int M = 64; // size of the cube
    const int STEPS = 50; // how many frames (files)
    const int TILE = 16;

    //before
    Kokkos::initialize();
    {
        Kokkos::View<double***> A ("A", M, M, M);//before
        Kokkos::View<double***> B ("B", M, M, M);//after

        Kokkos::deep_copy(A, 0.0);
        Kokkos::deep_copy(B, 0.0);//initalizing everything to 0


        //we'll make a hot spot in the middle of A to see how the heat spreads in B

        auto host_A1 = Kokkos::create_mirror_view(A);
        const int mid = M / 2;
        host_A1(mid, mid, mid) = 1.0e12;
        Kokkos::deep_copy(A, host_A1);


        auto host_A2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, A);
        std::vector<double> buf(M*M*M);
        std::copy(host_A2.data(), host_A2.data() + buf.size(), buf.begin());
        save_vtk(buf, M, "heat_000.vtk");

        const auto start = std::chrono::high_resolution_clock::now();


        using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Policy policy({1,1,1}, {M-1, M-1, M-1}, {TILE, TILE, TILE});

        auto t0 = std::chrono::high_resolution_clock::now();

        for (int t = 1; t <= STEPS; ++t) {
            Kokkos::parallel_for("HeatStep", policy, KOKKOS_LAMBDA (const int k, const int i, const int j) {
                B(k,i,j) = ( A(k,i,j) + A(k-1,i,j) + A(k+1,i,j) + A(k,i-1,j) + A(k,i+1,j) + A(k,i,j-1) + A(k,i,j+1) ) / 7.0;
            });
            Kokkos::fence();

            std::swap(A, B); //swapping to the new block after computing
            std::ostringstream name;
            name << "heat_" << std::setfill('0') << std::setw(3) << t << ".vtk";
            auto host_A3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, A);
            std::vector<double> buf(M*M*M);
            std::copy(host_A3.data(), host_A3.data() + buf.size(), buf.begin());
            save_vtk(buf, M, name.str());

        }


    const auto end = std::chrono::high_resolution_clock::now(); //measure computation

    auto result = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Computation took %lld nanoseconds", result);
    }
    Kokkos::finalize();
    return 0;
}


