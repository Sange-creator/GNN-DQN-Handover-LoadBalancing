// =========================================================================
// results.js — REAL data from eval_20seed (20 seeds, training complete)
// Source: results/runs/multiscenario_ue/eval_20seed/*.csv
// Each value: { mean, ci95 }. null = not reported.
// =========================================================================

const RESULT_STATUS = {
  state: "complete",
  message: "Training complete · 20-seed evaluation · numbers from results/runs/multiscenario_ue/eval_20seed/.",
  run: "multiscenario_ue",
  seeds: 20,
};

const METHOD_ORDER = [
  "no_handover", "random_valid", "strongest_rsrp",
  "a3_ttt", "load_aware", "gnn_dqn", "son_gnn_dqn",
];

// V(mean, ci95) → small object
const V = (m, c) => (m == null ? null : { m, c: c ?? 0 });

// Per scenario × method × metric. Metrics:
//   thr   = avg_ue_throughput_mbps
//   p5    = p5_ue_throughput_mbps
//   jain  = jain_load_fairness
//   pp    = pingpong_rate
//   ho    = handovers_per_1000_decisions
//   over  = overload_rate
//   out   = outage_rate
//   ldstd = load_std
//   son_n = son_update_count (only son_gnn_dqn)
//   son_c = son_avg_abs_cio_db
//   son_r = son_rollback_count
const RESULTS = {
  status: RESULT_STATUS,
  data: {
    dense_urban: {
      no_handover:    { thr:V(4.965,0.067), p5:V(2.348,0.059), jain:V(0.627,0.023), pp:V(0.000,0), ho:V(0.0,0),    over:V(0.0051,0.007), out:V(0,0), ldstd:V(0.225,0.012) },
      random_valid:   { thr:V(4.133,0.050), p5:V(1.946,0.045), jain:V(0.864,0.002), pp:V(0.052,0.001), ho:V(949.5,1.04), over:V(0.0011,0), out:V(0.00004,0), ldstd:V(0.150,0.002) },
      strongest_rsrp: { thr:V(4.929,0.060), p5:V(2.324,0.052), jain:V(0.537,0.011), pp:V(0.189,0.007), ho:V(35.8,1.08), over:V(0.0146,0.005), out:V(0,0), ldstd:V(0.264,0.006) },
      a3_ttt:         { thr:V(4.953,0.063), p5:V(2.343,0.055), jain:V(0.568,0.016), pp:V(0.000,0), ho:V(13.96,0.35), over:V(0.0078,0.006), out:V(0,0), ldstd:V(0.249,0.009) },
      load_aware:     { thr:V(4.937,0.061), p5:V(2.330,0.052), jain:V(0.540,0.013), pp:V(0.191,0.010), ho:V(27.77,0.76), over:V(0.0149,0.007), out:V(0,0), ldstd:V(0.263,0.007) },
      gnn_dqn:        { thr:V(4.971,0.061), p5:V(2.355,0.054), jain:V(0.668,0.025), pp:V(0.000,0), ho:V(4.52,0.11), over:V(0.0075,0.007), out:V(0,0), ldstd:V(0.213,0.013) },
      son_gnn_dqn:    { thr:V(4.953,0.063), p5:V(2.343,0.055), jain:V(0.566,0.016), pp:V(0.000,0), ho:V(13.98,0.37), over:V(0.0096,0.007), out:V(0,0), ldstd:V(0.250,0.009), son_n:V(18.5,1.71), son_c:V(0.023,0.002), son_r:V(1.45,0.18) },
    },
    highway: {
      no_handover:    { thr:V(4.896,0.101), p5:V(2.345,0.051), jain:V(0.816,0.028), pp:V(0,0), ho:V(0,0), over:V(0,0), out:V(0,0), ldstd:V(0.095,0.009) },
      random_valid:   { thr:V(4.050,0.082), p5:V(1.838,0.039), jain:V(0.793,0.005), pp:V(0.134,0.003), ho:V(875.9,2.17), over:V(0.0071,0.002), out:V(0.0178,0.001), ldstd:V(0.197,0.005) },
      strongest_rsrp: { thr:V(4.876,0.101), p5:V(2.334,0.051), jain:V(0.821,0.022), pp:V(0,0), ho:V(22.78,0.54), over:V(0,0), out:V(0,0), ldstd:V(0.077,0.006) },
      a3_ttt:         { thr:V(4.877,0.101), p5:V(2.334,0.051), jain:V(0.823,0.023), pp:V(0,0), ho:V(21.76,0.52), over:V(0,0), out:V(0,0), ldstd:V(0.076,0.006) },
      load_aware:     { thr:V(4.877,0.101), p5:V(2.334,0.051), jain:V(0.824,0.023), pp:V(0.003,0.004), ho:V(21.46,0.51), over:V(0,0), out:V(0,0), ldstd:V(0.076,0.006) },
      gnn_dqn:        { thr:V(4.889,0.101), p5:V(2.341,0.051), jain:V(0.656,0.032), pp:V(0.080,0.015), ho:V(7.55,0.79), over:V(0,0), out:V(0,0), ldstd:V(0.166,0.015) },
      son_gnn_dqn:    { thr:V(4.877,0.101), p5:V(2.335,0.051), jain:V(0.823,0.023), pp:V(0,0), ho:V(21.69,0.51), over:V(0,0), out:V(0,0), ldstd:V(0.076,0.006), son_n:V(19.15,1.81), son_c:V(0.096,0.009), son_r:V(2.2,0.20) },
    },
    suburban: {
      no_handover:    { thr:V(5.015,0.095), p5:V(2.363,0.066), jain:V(0.643,0.021), pp:V(0,0), ho:V(0,0), over:V(0.0042,0.006), out:V(0,0), ldstd:V(0.225,0.010) },
      random_valid:   { thr:V(4.150,0.072), p5:V(1.937,0.051), jain:V(0.836,0.002), pp:V(0.071,0.002), ho:V(933.7,1.07), over:V(0.0264,0.005), out:V(0.0016,0), ldstd:V(0.219,0.005) },
      strongest_rsrp: { thr:V(4.966,0.095), p5:V(2.336,0.064), jain:V(0.561,0.019), pp:V(0.218,0.009), ho:V(30.0,1.34), over:V(0.0107,0.008), out:V(0,0), ldstd:V(0.253,0.012) },
      a3_ttt:         { thr:V(4.998,0.097), p5:V(2.350,0.065), jain:V(0.586,0.020), pp:V(0,0), ho:V(9.87,0.45), over:V(0.0086,0.007), out:V(0,0), ldstd:V(0.243,0.012) },
      load_aware:     { thr:V(4.954,0.100), p5:V(2.329,0.063), jain:V(0.562,0.019), pp:V(0.380,0.013), ho:V(42.4,2.62), over:V(0.0112,0.007), out:V(0,0), ldstd:V(0.253,0.012) },
      gnn_dqn:        { thr:V(5.015,0.093), p5:V(2.361,0.066), jain:V(0.680,0.020), pp:V(0,0), ho:V(2.79,0.24), over:V(0.0023,0.004), out:V(0,0), ldstd:V(0.216,0.011) },
      son_gnn_dqn:    { thr:V(4.997,0.095), p5:V(2.350,0.064), jain:V(0.587,0.020), pp:V(0,0), ho:V(9.74,0.48), over:V(0.0087,0.007), out:V(0,0), ldstd:V(0.243,0.011), son_n:V(10.2,1.44), son_c:V(0.023,0.003), son_r:V(1.63,0.27) },
    },
    sparse_rural: {
      no_handover:    { thr:V(2.906,0.088), p5:V(1.098,0.128), jain:V(0.720,0.037), pp:V(0,0), ho:V(0,0), over:V(0,0), out:V(0.0394,0.012), ldstd:V(0.106,0.013) },
      random_valid:   { thr:V(2.453,0.075), p5:V(0.747,0.072), jain:V(0.753,0.010), pp:V(0.273,0.012), ho:V(722.0,15.8), over:V(0.0002,0), out:V(0.0676,0.010), ldstd:V(0.132,0.006) },
      strongest_rsrp: { thr:V(2.948,0.085), p5:V(1.246,0.100), jain:V(0.603,0.032), pp:V(0.241,0.034), ho:V(19.23,2.75), over:V(0,0), out:V(0.0214,0.010), ldstd:V(0.135,0.013) },
      a3_ttt:         { thr:V(2.934,0.086), p5:V(1.199,0.097), jain:V(0.633,0.030), pp:V(0,0), ho:V(4.93,0.55), over:V(0,0), out:V(0.0283,0.009), ldstd:V(0.127,0.012) },
      load_aware:     { thr:V(2.931,0.084), p5:V(1.202,0.101), jain:V(0.592,0.031), pp:V(0.393,0.036), ho:V(25.3,2.30), over:V(0,0), out:V(0.0251,0.010), ldstd:V(0.137,0.013) },
      gnn_dqn:        { thr:V(2.945,0.090), p5:V(1.204,0.116), jain:V(0.666,0.036), pp:V(0.020,0.018), ho:V(7.18,1.27), over:V(0,0), out:V(0.0253,0.011), ldstd:V(0.121,0.014) },
      son_gnn_dqn:    { thr:V(2.933,0.087), p5:V(1.198,0.106), jain:V(0.634,0.029), pp:V(0,0), ho:V(4.85,0.55), over:V(0,0), out:V(0.0284,0.011), ldstd:V(0.127,0.012), son_n:V(2.2,0.55), son_c:V(0.022,0.006), son_r:V(0.68,0.16) },
    },
    overloaded_event: {
      no_handover:    { thr:V(5.096,0.119), p5:V(2.654,0.101), jain:V(0.463,0.015), pp:V(0,0), ho:V(0,0), over:V(0.250,0), out:V(0,0), ldstd:V(0.660,0.019) },
      random_valid:   { thr:V(5.808,0.048), p5:V(3.574,0.045), jain:V(0.929,0.001), pp:V(0.090,0.001), ho:V(916.5,1.14), over:V(0.0594,0.005), out:V(0,0), ldstd:V(0.187,0.002) },
      strongest_rsrp: { thr:V(5.118,0.105), p5:V(2.568,0.106), jain:V(0.463,0.013), pp:V(0.200,0.014), ho:V(12.84,0.57), over:V(0.248,0.005), out:V(0,0), ldstd:V(0.658,0.018) },
      a3_ttt:         { thr:V(5.136,0.113), p5:V(2.609,0.112), jain:V(0.465,0.014), pp:V(0,0), ho:V(6.10,0.22), over:V(0.250,0.003), out:V(0,0), ldstd:V(0.656,0.019) },
      load_aware:     { thr:V(5.076,0.114), p5:V(2.600,0.088), jain:V(0.459,0.013), pp:V(0.007,0.005), ho:V(3.67,0.25), over:V(0.250,0.001), out:V(0,0), ldstd:V(0.664,0.018) },
      gnn_dqn:        { thr:V(4.873,0.100), p5:V(1.628,0.079), jain:V(0.384,0.016), pp:V(0,0), ho:V(9.95,0.18), over:V(0.226,0.020), out:V(0,0), ldstd:V(0.849,0.027) },
      son_gnn_dqn:    { thr:V(5.131,0.109), p5:V(2.607,0.107), jain:V(0.465,0.014), pp:V(0,0), ho:V(6.17,0.21), over:V(0.250,0.002), out:V(0,0), ldstd:V(0.656,0.019), son_n:V(45.2,1.05), son_c:V(0.157,0.004), son_r:V(3.0,0) },
    },
    real_pokhara: {
      no_handover:    { thr:V(4.985,0.060), p5:V(2.358,0.055), jain:V(0.827,0.010), pp:V(0,0), ho:V(0,0), over:V(0,0), out:V(0,0), ldstd:V(0.145,0.006) },
      random_valid:   { thr:V(3.924,0.040), p5:V(1.791,0.029), jain:V(0.867,0.001), pp:V(0.057,0.001), ho:V(945.7,0.89), over:V(0.166,0.010), out:V(0.0143,0), ldstd:V(0.282,0.003) },
      strongest_rsrp: { thr:V(4.967,0.060), p5:V(2.348,0.053), jain:V(0.780,0.017), pp:V(0.220,0.011), ho:V(20.26,0.63), over:V(0.0001,0), out:V(0,0), ldstd:V(0.160,0.009) },
      a3_ttt:         { thr:V(4.977,0.059), p5:V(2.354,0.054), jain:V(0.791,0.017), pp:V(0,0), ho:V(7.00,0.29), over:V(0.0023,0.002), out:V(0,0), ldstd:V(0.155,0.009) },
      load_aware:     { thr:V(4.965,0.060), p5:V(2.347,0.053), jain:V(0.780,0.014), pp:V(0.303,0.017), ho:V(23.32,1.35), over:V(0.0002,0), out:V(0,0), ldstd:V(0.160,0.008) },
      gnn_dqn:        { thr:V(4.589,0.088), p5:V(2.090,0.086), jain:V(0.510,0.026), pp:V(0.709,0.021), ho:V(63.3,5.52), over:V(0.0844,0.009), out:V(0,0), ldstd:V(0.386,0.022) },
      son_gnn_dqn:    { thr:V(4.978,0.059), p5:V(2.354,0.054), jain:V(0.792,0.016), pp:V(0,0), ho:V(7.21,0.32), over:V(0.0013,0.002), out:V(0,0), ldstd:V(0.155,0.009), son_n:V(48.0,0), son_c:V(0.060,0), son_r:V(3.0,0) },
    },
    pokhara_dense_peakhour: {
      no_handover:    { thr:V(4.093,0.023), p5:V(1.751,0.034), jain:V(0.895,0.009), pp:V(0,0), ho:V(0,0), over:V(0.813,0.020), out:V(0,0), ldstd:V(0.523,0.026) },
      random_valid:   { thr:V(1.526,0.004), p5:V(0.699,0.004), jain:V(0.960,0.001), pp:V(0.057,0.001), ho:V(944.8,0.44), over:V(1.000,0), out:V(0.0140,0), ldstd:V(0.705,0.005) },
      strongest_rsrp: { thr:V(4.111,0.037), p5:V(1.658,0.030), jain:V(0.843,0.010), pp:V(0.247,0.008), ho:V(20.42,0.57), over:V(0.726,0.020), out:V(0,0), ldstd:V(0.628,0.024) },
      a3_ttt:         { thr:V(4.131,0.029), p5:V(1.691,0.032), jain:V(0.855,0.010), pp:V(0,0), ho:V(6.34,0.15), over:V(0.748,0.027), out:V(0,0), ldstd:V(0.603,0.024) },
      load_aware:     { thr:V(4.105,0.030), p5:V(1.651,0.031), jain:V(0.841,0.009), pp:V(0.319,0.006), ho:V(22.53,0.60), over:V(0.724,0.020), out:V(0,0), ldstd:V(0.633,0.022) },
      gnn_dqn:        { thr:V(2.718,0.022), p5:V(0.604,0.026), jain:V(0.505,0.014), pp:V(0.751,0.009), ho:V(69.9,3.03), over:V(0.668,0.010), out:V(0.0001,0), ldstd:V(1.923,0.069) },
      son_gnn_dqn:    { thr:V(4.135,0.032), p5:V(1.693,0.036), jain:V(0.854,0.011), pp:V(0,0), ho:V(6.48,0.14), over:V(0.743,0.025), out:V(0,0), ldstd:V(0.604,0.026), son_n:V(48.0,0), son_c:V(0.060,0), son_r:V(3.0,0) },
    },
    kathmandu_real: {
      no_handover:    { thr:V(4.761,0.067), p5:V(2.215,0.040), jain:V(0.615,0.017), pp:V(0,0), ho:V(0,0), over:V(0.068,0.010), out:V(0.00003,0), ldstd:V(0.328,0.013) },
      random_valid:   { thr:V(3.720,0.022), p5:V(1.667,0.015), jain:V(0.882,0.001), pp:V(0.044,0.001), ho:V(957.1,0.90), over:V(0.330,0.014), out:V(0.0122,0), ldstd:V(0.320,0.004) },
      strongest_rsrp: { thr:V(4.652,0.058), p5:V(2.130,0.038), jain:V(0.567,0.017), pp:V(0.243,0.008), ho:V(32.36,0.91), over:V(0.0703,0.006), out:V(0,0), ldstd:V(0.345,0.013) },
      a3_ttt:         { thr:V(4.707,0.065), p5:V(2.180,0.037), jain:V(0.581,0.016), pp:V(0,0), ho:V(7.99,0.33), over:V(0.0664,0.007), out:V(0,0), ldstd:V(0.338,0.012) },
      load_aware:     { thr:V(4.652,0.066), p5:V(2.136,0.038), jain:V(0.569,0.018), pp:V(0.344,0.008), ho:V(36.74,1.45), over:V(0.0693,0.006), out:V(0,0), ldstd:V(0.344,0.013) },
      gnn_dqn:        { thr:V(2.678,0.052), p5:V(0.687,0.027), jain:V(0.209,0.007), pp:V(0.454,0.023), ho:V(49.73,2.25), over:V(0.184,0.006), out:V(0.0001,0), ldstd:V(1.055,0.033) },
      son_gnn_dqn:    { thr:V(4.704,0.064), p5:V(2.172,0.037), jain:V(0.582,0.016), pp:V(0,0), ho:V(8.60,0.29), over:V(0.069,0.007), out:V(0,0), ldstd:V(0.337,0.012), son_n:V(48.0,0), son_c:V(0.0384,0), son_r:V(3.0,0) },
    },
    dharan_synthetic: {
      no_handover:    { thr:V(5.079,0.079), p5:V(2.329,0.043), jain:V(0.675,0.028), pp:V(0,0), ho:V(0,0), over:V(0,0), out:V(0,0), ldstd:V(0.143,0.009) },
      random_valid:   { thr:V(4.166,0.063), p5:V(1.876,0.038), jain:V(0.802,0.002), pp:V(0.054,0.001), ho:V(947.5,1.18), over:V(0.0151,0.002), out:V(0.0085,0.001), ldstd:V(0.218,0.003) },
      strongest_rsrp: { thr:V(5.062,0.079), p5:V(2.322,0.043), jain:V(0.599,0.015), pp:V(0.212,0.012), ho:V(18.71,0.88), over:V(0,0), out:V(0,0), ldstd:V(0.160,0.005) },
      a3_ttt:         { thr:V(5.072,0.079), p5:V(2.325,0.043), jain:V(0.621,0.018), pp:V(0,0), ho:V(7.03,0.36), over:V(0,0), out:V(0,0), ldstd:V(0.154,0.006) },
      load_aware:     { thr:V(5.056,0.079), p5:V(2.315,0.043), jain:V(0.594,0.017), pp:V(0.340,0.019), ho:V(24.59,1.47), over:V(0,0), out:V(0,0), ldstd:V(0.162,0.006) },
      gnn_dqn:        { thr:V(4.994,0.071), p5:V(2.301,0.044), jain:V(0.452,0.016), pp:V(0.428,0.082), ho:V(17.99,3.46), over:V(0.0177,0.009), out:V(0,0), ldstd:V(0.253,0.012) },
      son_gnn_dqn:    { thr:V(5.072,0.079), p5:V(2.325,0.043), jain:V(0.617,0.018), pp:V(0,0), ho:V(7.07,0.32), over:V(0,0), out:V(0,0), ldstd:V(0.155,0.006), son_n:V(46.95,0.79), son_c:V(0.0587,0.001), son_r:V(3.0,0) },
    },
    unknown_hex_grid: {
      no_handover:    { thr:V(4.979,0.047), p5:V(2.328,0.060), jain:V(0.627,0.023), pp:V(0,0), ho:V(0,0), over:V(0.0081,0.006), out:V(0,0), ldstd:V(0.234,0.012) },
      random_valid:   { thr:V(4.109,0.034), p5:V(1.909,0.045), jain:V(0.835,0.002), pp:V(0.055,0.001), ho:V(947.4,0.82), over:V(0.0264,0.003), out:V(0.0020,0), ldstd:V(0.222,0.003) },
      strongest_rsrp: { thr:V(4.903,0.048), p5:V(2.289,0.057), jain:V(0.525,0.011), pp:V(0.196,0.009), ho:V(34.23,1.47), over:V(0.0228,0.006), out:V(0,0), ldstd:V(0.273,0.007) },
      a3_ttt:         { thr:V(4.935,0.046), p5:V(2.308,0.058), jain:V(0.553,0.015), pp:V(0,0), ho:V(11.61,0.42), over:V(0.0232,0.006), out:V(0,0), ldstd:V(0.262,0.008) },
      load_aware:     { thr:V(4.886,0.051), p5:V(2.275,0.054), jain:V(0.520,0.016), pp:V(0.348,0.009), ho:V(48.51,3.15), over:V(0.0240,0.006), out:V(0,0), ldstd:V(0.276,0.010) },
      gnn_dqn:        { thr:V(4.983,0.045), p5:V(2.330,0.058), jain:V(0.669,0.019), pp:V(0,0), ho:V(3.08,0.18), over:V(0.0027,0.004), out:V(0,0), ldstd:V(0.223,0.010) },
      son_gnn_dqn:    { thr:V(4.933,0.042), p5:V(2.308,0.058), jain:V(0.555,0.017), pp:V(0,0), ho:V(11.67,0.40), over:V(0.0231,0.006), out:V(0,0), ldstd:V(0.261,0.009), son_n:V(15.8,1.96), son_c:V(0.0219,0.003), son_r:V(1.8,0.24) },
    },
    coverage_hole: {
      no_handover:    { thr:V(4.981,0.136), p5:V(2.402,0.104), jain:V(0.773,0.034), pp:V(0,0), ho:V(0,0), over:V(0,0), out:V(0,0), ldstd:V(0.091,0.010) },
      random_valid:   { thr:V(4.197,0.114), p5:V(2.006,0.086), jain:V(0.841,0.002), pp:V(0.139,0.003), ho:V(875.0,2.62), over:V(0,0), out:V(0,0), ldstd:V(0.080,0.002) },
      strongest_rsrp: { thr:V(4.965,0.135), p5:V(2.393,0.102), jain:V(0.773,0.031), pp:V(0.120,0.028), ho:V(17.67,1.22), over:V(0,0), out:V(0,0), ldstd:V(0.091,0.009) },
      a3_ttt:         { thr:V(4.971,0.135), p5:V(2.396,0.103), jain:V(0.774,0.032), pp:V(0,0), ho:V(11.38,0.61), over:V(0,0), out:V(0,0), ldstd:V(0.090,0.010) },
      load_aware:     { thr:V(4.974,0.136), p5:V(2.398,0.103), jain:V(0.767,0.030), pp:V(0,0), ho:V(7.89,0.45), over:V(0,0), out:V(0,0), ldstd:V(0.092,0.009) },
      gnn_dqn:        { thr:V(4.981,0.136), p5:V(2.402,0.104), jain:V(0.773,0.034), pp:V(0,0), ho:V(0,0), over:V(0,0), out:V(0,0), ldstd:V(0.091,0.010) },
      son_gnn_dqn:    { thr:V(4.971,0.135), p5:V(2.396,0.103), jain:V(0.774,0.032), pp:V(0,0), ho:V(11.38,0.61), over:V(0,0), out:V(0,0), ldstd:V(0.090,0.010), son_n:V(0,0), son_c:V(0,0), son_r:V(0,0) },
    },
  },
};

// ====== Training/topology curves left as illustrative (unchanged shape) ====
const TRAINING_CURVES = {
  episodes: 300,
  loss: Array.from({ length: 300 }, (_, i) => {
    const e = i + 1;
    return Math.max(0.05, 0.9 * Math.exp(-e / 60) + 0.08 + (Math.sin(e * 0.5) + Math.sin(e * 1.3) * 0.6) * 0.015);
  }),
  reward: Array.from({ length: 300 }, (_, i) => {
    const e = i + 1;
    return -1.2 + 2.0 * (1 - Math.exp(-e / 90)) + (Math.sin(e * 0.7) + Math.sin(e * 1.9) * 0.4) * 0.06;
  }),
  epsilon: Array.from({ length: 300 }, (_, i) => Math.max(0.03, Math.exp(-(i + 1) / 90))),
  status: "complete",
};

// =========================================================================
// Head-to-head comparison: SON-GNN-DQN vs every baseline.
// Computed at module load. For each scenario × metric we record where the
// proposed method wins, ties, or loses, and by how much.
// =========================================================================

const KEY_METRICS = [
  { id: "thr",  label: "Avg UE throughput", unit: "Mbps", higher: true },
  { id: "p5",   label: "P5 throughput",     unit: "Mbps", higher: true },
  { id: "jain", label: "Jain fairness",     unit: "",     higher: true },
  { id: "pp",   label: "Ping-pong rate",    unit: "",     higher: false },
  { id: "ho",   label: "Handovers / 1k",    unit: "",     higher: false },
  { id: "over", label: "Overload rate",     unit: "",     higher: false },
];

function computeComparisons() {
  const out = {};
  const scenarios = Object.keys(RESULTS.data);
  for (const sid of scenarios) {
    const row = RESULTS.data[sid];
    const ours = row.son_gnn_dqn;
    if (!ours) continue;
    out[sid] = {};
    for (const m of KEY_METRICS) {
      const o = ours[m.id]?.m;
      out[sid][m.id] = {};
      for (const bid of METHOD_ORDER) {
        if (bid === "son_gnn_dqn") continue;
        const b = row[bid]?.[m.id]?.m;
        if (o == null || b == null) { out[sid][m.id][bid] = { verdict: "n/a" }; continue; }
        const delta = o - b;
        const better = m.higher ? delta >= -1e-6 : delta <= 1e-6;
        const sig = Math.abs(delta) > Math.max(Math.abs(b) * 0.01, 1e-4);
        out[sid][m.id][bid] = {
          delta,
          pct: b !== 0 ? (delta / b) * 100 : null,
          verdict: !sig ? "tie" : (better ? "win" : "loss"),
        };
      }
    }
  }
  return out;
}

const COMPARISONS = computeComparisons();

// =========================================================================
// Aggregations for the dashboard cards.
// =========================================================================

const SCENARIO_IDS = Object.keys(RESULTS.data);

function meanAcrossScenarios(metric) {
  // Returns [{ method, value, ci }] averaged across all scenarios.
  return METHOD_ORDER.map((mid) => {
    let sum = 0, cisum = 0, n = 0;
    for (const sid of SCENARIO_IDS) {
      const cell = RESULTS.data[sid]?.[mid]?.[metric];
      if (cell && cell.m != null) { sum += cell.m; cisum += cell.c || 0; n++; }
    }
    return { method: mid, value: n ? sum / n : null, ci: n ? cisum / n : 0 };
  }).filter((r) => r.value != null);
}

RESULTS.throughput = meanAcrossScenarios("thr");
RESULTS.pingpong   = meanAcrossScenarios("pp").map((r) => ({ ...r, value: r.value * 100, ci: r.ci * 100 }));
RESULTS.jain       = meanAcrossScenarios("jain");
RESULTS.outage     = meanAcrossScenarios("out").map((r) => ({ ...r, value: r.value * 100, ci: r.ci * 100 }));

// Radar: normalize each axis across methods to 0..1 (higher = better).
const RADAR_AXES = [
  { id: "thr",  invert: false },
  { id: "p5",   invert: false },
  { id: "jain", invert: false },
  { id: "pp",   invert: true  }, // stability
  { id: "out",  invert: true  }, // coverage
  { id: "ho",   invert: true  }, // efficiency (lower handovers)
];
RESULTS.radar = (function () {
  const avgs = {};
  METHOD_ORDER.forEach((mid) => {
    avgs[mid] = RADAR_AXES.map((ax) => {
      const m = meanAcrossScenarios(ax.id).find((r) => r.method === mid);
      return m?.value ?? 0;
    });
  });
  // Normalize per-axis to 0..1.
  return METHOD_ORDER.map((mid) => {
    const values = avgs[mid].map((v, i) => {
      const all = METHOD_ORDER.map((m) => avgs[m][i]);
      const lo = Math.min(...all), hi = Math.max(...all);
      if (hi - lo < 1e-9) return 0.5;
      const norm = (v - lo) / (hi - lo);
      return RADAR_AXES[i].invert ? 1 - norm : norm;
    });
    return { method: mid, values };
  });
})();

// Topology generalization: throughput vs scenario cell count (approx).
RESULTS.generalization = (function () {
  const pairs = SCENARIO_IDS.map((sid) => {
    const s = window.SCENARIO_BY_ID[sid];
    return { cells: s?.cells || 10, sid };
  }).sort((a, b) => a.cells - b.cells);
  const series = ["son_gnn_dqn", "gnn_dqn", "a3_ttt"].map((mid) => ({
    method: mid,
    points: pairs.map((p) => [p.cells, RESULTS.data[p.sid]?.[mid]?.thr?.m]).filter(([, y]) => y != null),
  }));
  return series;
})();

// CIO histogram: synthesized from per-scenario son_c (avg abs CIO dB).
RESULTS.cio_histogram = (function () {
  const counts = new Array(13).fill(0); // bins for 0..6 dB in 0.5 dB steps
  for (const sid of SCENARIO_IDS) {
    const cell = RESULTS.data[sid]?.son_gnn_dqn;
    if (!cell?.son_n?.m || !cell?.son_c?.m) continue;
    const idx = Math.min(12, Math.round(cell.son_c.m / 0.5));
    counts[idx] += cell.son_n.m;
  }
  return counts.map((c, i) => ({
    count: c,
    label: (i * 0.5).toFixed(1),
    min: i * 0.5,
    max: (i + 1) * 0.5,
  }));
})();

// Training curves: shape into series.
RESULTS.training_curves = [
  { method: "son_gnn_dqn", label: "Reward", points: TRAINING_CURVES.reward.map((y, i) => [i, (y + 1.2) / 2.0]) },
  { method: "a3_ttt", label: "Loss", color: "oklch(0.65 0.150 25)", points: TRAINING_CURVES.loss.map((y, i) => [i, y]) },
];

// Per-scenario throughput table for ResultsTable.
RESULTS.per_scenario = SCENARIO_IDS.map((sid) => {
  const s = window.SCENARIO_BY_ID[sid];
  const values = {};
  METHOD_ORDER.forEach((mid) => {
    values[mid] = RESULTS.data[sid]?.[mid]?.thr?.m ?? null;
  });
  return { scenario: s?.name || sid, sid, values };
});

window.RESULTS = RESULTS;
window.METHOD_ORDER = METHOD_ORDER;
window.KEY_METRICS = KEY_METRICS;
window.COMPARISONS = COMPARISONS;
window.TRAINING_CURVES = TRAINING_CURVES;
