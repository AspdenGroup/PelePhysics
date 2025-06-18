#ifndef TURBFORCING_H
#define TURBFORCING_H

#include <AMReX_ParmParse.H>
#include <AMReX_Arena.H>
#include <depRand.H>

namespace pele::physics::turbforcing {

struct TurbForcingParm
{
  int verbose = 0;
  // factor by which to reduce sampling for faster performance
  int ff_factor = 4;
  // how many modes to use
  int nmodes = 4;
  // don't use any modes below mode_start. We probably don't need this
  int mode_start = 0;
  // fine tune force scale
  amrex::Real force_scale_fudge = 1.0;
  // suppress symmetry breaking modes
  amrex::Real forcing_epsilon = 0.1;
  // time offset for forcing function
  amrex::Real time_offset = 0.0;
  amrex::Real urms;
  int spectrum_type = 2;
  int moderate_zero_modes = 1;
  int hack_lz = 0;
  constexpr int array_size = 33;
  constexpr int num_fdarray = 17;

  //AJA's reference values for forcing scales
  constexpr amrex::Real flr = 2.64e-2;
  constexpr amrex::Real fvr = 3.27;
  constexpr amrex::Real fts_min = 1.21e-3;
  constexpr amrex::Real fts_max = 2.42e-3;
  amrex::Real fsr;
  //computed values for forcing
  amrex::Real force_scale;
  amrex::Real forcing_time_scale_min;
  amrex::Real forcing_time_scale_max;
  amrex::Real Lmin;
  amrex::Real kappaMax;
  int nxmodes;
  int nymodes;
  int nzmodes;
  amrex::Real freqMin;
  amrex::Real freqMax;
  amrex::Real freqDiff;
  
  // forcedata will contain num_fdarray arrays of size (0,0,0)(array_size-1,array_size-1,array_size-1)
  amrex::Real* forcedata;
}
  
struct TurbForcing
{
public:
  TurbForcing() = default;

  ~TurbForcing() = default;
  
  // function to generate the parameters
  void init(amrex::GeometryData const& geomdata);

  // function to make the forcing term
  void addTurbVelForces(amrex::GeometryData const& geomdata,
			  const amrex::Box& bx,
			  const amrex::Real& a_time,
			  amrex::Array4<amrex::Real> const& force,
			  amrex::Array4<const amrex::Real> const& rho);

  void deriveForcing(PeleLM* a_pelelm,
		       const amrex::Box& bx,
		       amrex::FArrayBox& derfab,
		       int dcomp,
		       int ncomp,
		       const amrex::FArrayBox& statefab,
		       const amrex::FArrayBox& reactfab,
		       const amrex::FArrayBox& pressfab,
		       const amrex::Geometry& geom,
		       amrex::Real time,
		       const amrex::Vector<amrex::BCRec>& bcrec,
		       int level);
private:
  TurbForcingParm tfp;
  bool turbforcing_initialized = false;
};

} //namespace pele::physics::turbforcing
#endif
