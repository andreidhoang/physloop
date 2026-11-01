# S1 finite-sample decision argument

This is an agent-authored mathematical review of the autonomous S1 implementation dated 2026-10-31. It describes the conservative decision method in `S1_IMPLEMENTATION.md`, including its assumptions and limits. It is not evidence of Huy's independent derivation or a claim about real driving.

## What the gate estimates

Let a positive difference mean that candidate B is worse than reference A. Average paired replicate differences within a scene, scene differences within a road, and road differences within a stratum. The result targets the expected difference for an equally weighted road, then an equally weighted scene within that road, then an equally weighted scheduled replicate within that scene.

This equal-road estimand is not an equal-scene mean when roads have different scene counts. It is also not a per-distance event rate. Repeating a rollout does not create a new independent road. Within-road dependence may be arbitrary, provided that the complete road average remains in [-1,1].

For the stated population interpretation, complete roads are independent draws from a fixed stratum population. The road and scene membership, metric direction, positive margin, replicate schedule, required strata, look schedule and allocation are fixed before seeing outcomes. No road crosses strata in the implementation. Once a road enters a look its observations are complete: appending more scenes or replicates would change an already counted observation.

The fixed-look inequalities also cover independent, nonidentically distributed bounded observations relative to their average expected value. That extension does not by itself justify comparing successive looks with different target populations. The operational contract therefore keeps a stable stratum population and estimand across looks.

VI: Đơn vị độc lập là road, không phải số dòng dữ liệu. Nếu một road có nhiều scene hơn, nó vẫn chỉ có một phiếu trong trung bình này. Đổi sang trọng số theo scene là đổi câu hỏi thống kê và cần kiểm định lại thiết kế.

## A bounded observation need not be Bernoulli

For independent observations Z_i in [0,1], let zbar be their empirical mean and mu their common mean. Convexity gives, for any real lambda,

    E exp(lambda Z_i) <= 1 - mu + mu exp(lambda).

Multiplying this bound across independent observations and optimizing the exponential Markov bound gives

    P(zbar >= x) <= exp(-R kl(x, mu)),  x > mu,
    P(zbar <= x) <= exp(-R kl(x, mu)),  x < mu,

where R is the number of independent roads and

    kl(x, q) = x log(x/q) + (1-x) log((1-x)/(1-q)).

Endpoint expressions use their continuous limits. The corresponding inversion defines L(zbar,R,t) and U(zbar,R,t) from kl(zbar,q) <= log(1/t)/R, with endpoints 0 and 1 retained. Each is a separate one-sided bound with failure probability at most t. Applying Bernoulli KL here does not assert that a fractional road average is a binary observation. The argument uses its bounded range.

The primary foundation is [Hoeffding (1963), Probability Inequalities for Sums of Bounded Random Variables](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf). A direct treatment of inversion for unit-interval observations is [Foong, Bruinsma and Burt (2022), A Note on the Chernoff Bound for Random Variables in the Unit Interval](https://arxiv.org/abs/2205.07880).

## Paired road differences

For road difference d_r, define X_r=max(d_r,0) and Y_r=max(-d_r,0). Both lie in [0,1], and delta=E[d_r]=E[X_r]-E[Y_r]. The two quantities within a road are dependent; independence between them is not needed.

At directional error budget a, set

    lower_delta = L(mean(X), R, a/2) - U(mean(Y), R, a/2),
    upper_delta = U(mean(X), R, a/2) - L(mean(Y), R, a/2).

If both component bounds defining upper_delta cover, upper_delta >= delta. By a union bound the failure probability is at most a/2+a/2=a. The lower bound has the same separate guarantee. These are two one-sided bounds; displaying them together does not create a two-sided interval with coverage 1-a. A simultaneous two-sided claim would require an additional error allocation.

The implementation uses outward numerical rounding to avoid turning a numerical root error into an inward confidence limit. Exhaustive small-support tests inspect the implemented inversion independently of its formulas.

When all observed d_r are zero, upper_delta before outward rounding is 1-(a/2)^(1/R). At a=0.05 this is approximately 0.336 for nine roads and 0.0166 for 220 roads. Thus the existing nine-road all-zero example is INSUFFICIENT at margin 0.05 while a large all-zero stratum can establish GO. There is no universal nine-scene prohibition: the inference depends on the margin, outcomes, independent-road count and error budget.

VI: Giới hạn trên không bằng 0 dù mọi chênh lệch đã thấy đều bằng 0. Dữ liệu hữu hạn vẫn có thể chưa gặp một road làm lộ hồi quy. Số road độc lập tăng thì phần bất định này giảm.

## GO, regression flags and multiplicity

For stratum s with positive margin Delta_s, the non-inferiority null is delta_s >= Delta_s. GO requires upper_delta_s < Delta_s for every required stratum. If any stratum is truly at or beyond its margin, choose one such stratum s*. A false overall GO requires a false efficacy decision in s*. Its probability at a fixed look is at most a. No across-strata Bonferroni adjustment is required for this intersection-union decision, and the proof does not require independent strata. Missing required strata remain INSUFFICIENT and block GO.

The probability of overall GO equals the product of per-stratum powers only when the stratum decisions are independent. Road-disjoint membership alone is not an empirical demonstration of independence. Canonical product-power tables state their independence assumption separately.

Regression flags answer a different question: is delta_s > 0? Hoeffding's bounded-difference inequality gives the valid one-sided p-value

    p_s = min(1, exp(-R_s * max(mean(d_s),0)^2 / 2)).

The factor 1/2 follows from the width-two range [-1,1]. Under delta_s <= 0, the chance that p_s <= u is at most u. Holm's step-down procedure controls the familywise regression-flag error using these valid p-values even when strata are dependent. Keep the original registered family; inactive strata use p=1 instead of shrinking the family based on outcomes. The primary procedure is [Holm (1979), A Simple Sequentially Rejective Multiple Test Procedure](https://www.jstor.org/stable/4615733).

Regression flags take precedence over NI success. Consequently, a precisely detected small positive regression can be NO-GO even when it lies inside the NI margin. This is intentional in the implementation contract. The false-GO guarantee and the regression-family guarantee are separate error budgets; they are not a combined alpha bound over every possible error type.

## Finite, predeclared sequential looks

Let nonnegative weights w_1,...,w_K sum to at most one. Look k uses a_k=alpha*w_k. For any truly non-inferior-null stratum s*, the chance of an erroneous efficacy crossing at any scheduled look is at most sum(a_k)<=alpha. Overall GO implies such a crossing for s*, even if different strata finish at different looks. Stopping completed strata therefore does not invalidate this bound. Similarly, Holm family error at look k is at most a_k, and the chance of any false regression flag over the K looks is at most alpha.

This proof does not assume independent looks. It does require that every possible examination of outcomes is included in the predetermined schedule, with complete roads and a stable estimand. Choosing when to look, which new roads to sample, or which observations to include based on emerging results is not covered. Prefixes and new batches must satisfy the recorded admission and state-transition rules. An efficacy-completed stratum cannot receive new observations. The implementation must reject replayed looks, altered prior observations and observations after a terminal verdict.

A predeclared futility rule can only remove future opportunities for GO, so it cannot increase false-GO. Futility returns INSUFFICIENT, except when a regression flag already requires NO-GO. The no-early-futility schedule used in the synthetic experiment leaves unfinished strata INSUFFICIENT at the last look.

The displayed spent-alpha bounds remain valid under this finite-look union argument after stopping. They are not stage-wise-ordering intervals. The classical O'Brien-Fleming, Pocock and Lan-DeMets normal boundary tables are planning references with their own canonical joint-normal model; they do not authorize this road-bounded operational gate or predict its power.

VI: Mỗi lần nhìn kết quả tiêu một phần ngân sách sai số đã ghi trước. Không cần các lần nhìn độc lập, nhưng không được thêm lần nhìn hoặc đổi cách lấy road sau khi đã thấy kết quả. Các bảng biên chuẩn tắc và verifier bảo thủ là hai phương pháp khác nhau.

## What the evidence can establish

The argument above is a conditional mathematical guarantee, subject to sampling assumptions and correct implementation. Exhaustive small-support checks sum known binomial and multinomial probabilities without Monte Carlo randomness. They can expose implementation errors but do not replace the proof or cover every possible distribution.

A 1,000-trial Monte Carlo estimates behavior on each registered synthetic population. Its false-GO count needs binomial uncertainty, including the preregistered simultaneous acceptance rule. An observed fraction below alpha alone is not proof that the population probability is below alpha. Conversely, matching the printed historical power examples does not establish calibration of the new operational gate.

Equal candidate/reference behavior with a positive NI margin is an acceptable alternative, not the false-GO null. Harm-boundary scenarios must set at least one true stratum difference at or above its actual margin. Rare-stratum scenarios test whether a pooled average hides a harmful subgroup. Removing stratification changes the target question: a pooled GO can be correct for the pooled target while violating the original all-strata requirement. Label that outcome a GO that fails the original all-strata requirement, not a type-I error for the pooled hypothesis.

Finite-sample error control does not imply useful power. Report actual operational GO, NO-GO and INSUFFICIENT frequencies alongside the separate planning formulas. Rate-ratio planning and Tango/road-bootstrap diagnostic intervals are not finite-sample authorization proofs for this verifier.

## Boundary before S2

S1 qualifies synthetic software behavior under the stated design. S2 still needs to qualify road independence and clustering, the target sampling population, strata and weights, actual seed behavior, replicate dependence, evaluator identity, metric direction and deviation convention. Counting >=4 m outcomes separately while including them under a declared convention is transparent bookkeeping; it does not establish that rendered observations beyond the envelope measure policy quality faithfully.

Until these assumptions and protocol choices are justified, neither the proof nor a passed Monte Carlo suite licenses real-model conclusions. S1 also does not resolve publication licenses, spending, hardware approvals, real-world validity or Huy's independent mastery requirements.

VI: S1 kiểm tra dụng cụ trên thế giới tổng hợp mà ta biết luật sinh dữ liệu. S2 phải kiểm tra xem dữ liệu thật của simulator có đáp ứng các giả định đó hay không. Chứng minh đúng dưới giả định không tự chứng minh giả định đúng ngoài thực nghiệm.
