<%
cfg['compiler_args'] = ['-std=c++14', '-fopenmp', '-O3']
cfg['linker_args'] = ['-lgomp','-lgmp', '-lflint-arb', '-lflint', '-lmpfr']
cfg['include_dirs'] = ['-I/usr/include/flint']
setup_pybind11(cfg)
%>

#include <arb_hypgeom.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>


#include<stdio.h>

namespace py = pybind11;

py::array_t<double> hypergeometric_pfq(
        py::array_t<double> avi
        , py::array_t<double> bvi
        , py::array_t<double> zi
        , long precision
        ) {

    flint_rand_t state;

    flint_randinit(state);

    slong prec;
    prec = (slong)precision;

    arb_ptr av, bv;

    auto avx = avi.unchecked<2>();
    auto bvx = bvi.unchecked<2>();
    auto zx = zi.unchecked<1>();

    ssize_t d_a = avx.shape(1),
            d_b = bvx.shape(1);
    av = _arb_vec_init(d_a);
    bv = _arb_vec_init(d_b);

    auto call = [=](int idx) -> double{
        arb_t z, r;
        arb_init(z);
        arb_init(r);
//        fprintf(stderr,"%d ",idx);
        for(ssize_t i = 0; i < avx.shape(1); i++){
            arb_set_d(av+i, avx(idx,i));
        }

        for(ssize_t i = 0; i < bvx.shape(1); i++){
            arb_set_d(bv+i, bvx(idx,i));
        }
        arb_set_d(z, zx(idx));
        arb_hypgeom_pfq(r, av, d_a, bv, d_b, z, 0, prec);
        return arf_get_d(arb_midref(r), ARF_RND_NEAR);
    };
    auto result = py::array_t<double>(avx.shape(0));
    py::buffer_info buf3 = result.request();
    double *ptr3 = (double *) buf3.ptr;
    #pragma omp parallel for
    for(int i = 0; i < avx.shape(0); i++){

        ptr3[i] = call(i);
    }

    _arb_vec_clear(av, d_a) ;
    _arb_vec_clear(bv, d_b) ;
    flint_randclear(state);
    flint_cleanup();
    return result;
}

PYBIND11_MODULE(arb_wrapper, m) {
    m.def("hypergeometric_pfq_cpu", &hypergeometric_pfq, "hypergeometric_pfq");
}
