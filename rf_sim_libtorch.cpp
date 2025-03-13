#include <torch/extension.h>
#include <vector> 
#include <cmath> 
#include <tuple>

torch::Tensor simulate_longitudinal_magnetization_libtorch(
    torch::Tensor rf,
    torch::Tensor freq,
    torch::Tensor b1,
    float dt
)
{

    // get dimensions 
    auto nrf = rf.numel();
    auto nfreq = freq.numel();
    auto nb1 = b1.numel();

    // initialize magnetization at thermal equilibrium
    std::vector<int64_t> dims{nfreq, nb1};
    torch::Tensor mx = torch::zeros(dims, torch::dtype(torch::kDouble).device(rf.device()));
    torch::Tensor my = torch::zeros(dims, torch::dtype(torch::kDouble).device(rf.device()));
    torch::Tensor mz = torch::ones(dims, torch::dtype(torch::kDouble).device(rf.device()));

    // loop over points in RF pulse 
    for (int n=0; n<nrf; n++) {

        // x rotation for real part of RF 
        auto theta_x = (2 * M_PI * dt) * torch::real(rf.index({n})).view({1,1}) * b1.view({1,nb1});
        auto y1 = my*torch::cos(theta_x) - mz*torch::sin(theta_x);
        auto z1 = my*torch::sin(theta_x) + mz*torch::cos(theta_x);
        my = y1;
        mz = z1;

        // y rotation for imaginary part of RF 
        auto theta_y = (2 * M_PI * dt) * torch::imag(rf.index({n})).view({1,1}) * b1.view({1,nb1});
        auto x2 = mx*torch::cos(theta_y) + mz*torch::sin(theta_y);
        auto z2 = -mx*torch::sin(theta_y) + mz*torch::cos(theta_y);
        mx = x2;
        mz = z2;

        // z rotation for off-resonance 
        auto theta_z = (2 * M_PI * dt) * freq.view({nfreq, 1});
        auto x3 = mx*torch::cos(theta_z) - my*torch::sin(theta_z);
        auto y3 = mx*torch::sin(theta_z) + my*torch::cos(theta_z);
        mx = x3;
        my = y3;

    }

    return mz;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("simulate_longitudinal_magnetization_libtorch", &simulate_longitudinal_magnetization_libtorch, "mz-simulator for adiabatic inversion pulses");
}
