# S1 synthetic qualification results

Experiment pass: **True**. Registered qualification: **True**. CPU only; cloud cost $0.

The operational method is the conservative road-level KL gate, not the Gaussian planning design. Each endpoint is separately one-sided. Raw trials and source hashes are adjacent to this report.

| Cell | GO | NO-GO | INSUFFICIENT | Error upper bound |
|---|---:|---:|---:|---:|
| cluster_boundary/fixed | 0 | 0 | 1000 | 0.005974 |
| cluster_boundary/sequential | 0 | 0 | 1000 | 0.005974 |
| cluster_harmful/fixed | 0 | 0 | 1000 | 0.005974 |
| cluster_harmful/sequential | 0 | 0 | 1000 | 0.005974 |
| cluster_one_bad/fixed | 0 | 0 | 1000 | 0.005974 |
| cluster_one_bad/sequential | 0 | 0 | 1000 | 0.005974 |
| iid_boundary/fixed | 0 | 0 | 1000 | 0.005974 |
| iid_boundary/sequential | 0 | 0 | 1000 | 0.005974 |
| iid_equivalent/fixed | 0 | 0 | 1000 | 0.005974 |
| iid_equivalent/sequential | 1 | 0 | 999 | 0.005974 |
| iid_gross/fixed | 0 | 1000 | 0 | 0.005974 |
| iid_gross/sequential | 0 | 999 | 1 | 0.005974 |
| iid_harmful/fixed | 0 | 0 | 1000 | 0.005974 |
| iid_harmful/sequential | 0 | 0 | 1000 | 0.005974 |
| iid_identical/fixed | 1000 | 0 | 0 | 0.005974 |
| iid_identical/sequential | 1000 | 0 | 0 | 0.005974 |
| iid_one_bad/fixed | 3 | 0 | 997 | 0.011835 |
| iid_one_bad/sequential | 2 | 0 | 998 | 0.010084 |
| rare_harm/fixed | 0 | 0 | 1000 | 0.005974 |
| rare_harm/pooled_fixed | 999 | 0 | 1 | not the same estimand |
| rare_harm/sequential | 0 | 0 | 1000 | 0.005974 |

The error bounds use exact binomial inversion with a simultaneous 95% family over 20 registered calibration cells. Monte Carlo supports only these configurations; the mathematical guarantee relies on independent road clusters and a fixed sampling design.

Pooled rare-harm GO is not a false-GO for the pooled estimand. It demonstrates why a pooled claim cannot replace the registered per-stratum claim.

Gaussian power tables, IID Tango and independent-Poisson rate-ratio calculations are planning/reference outputs. S2 must measure actual dependence and score variability.

VI: Các số trên thuộc dữ liệu giả với sự thật đã biết. Không phải kết quả của Alpamayo hoặc bằng chứng lái xe ngoài đời.
