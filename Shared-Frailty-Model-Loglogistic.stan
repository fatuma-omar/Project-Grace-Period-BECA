functions{
  real loglogistic_hazard(real t, real linear_predictor, real log_lambda, real log_k){
    real lambda = exp(log_lambda);
    real k = exp(log_k);
    return exp(linear_predictor)*k*lambda*pow(lambda*t, k - 1)/(1 + pow(lambda*t, k));
  }

  real loglogistic_cumulative_hazard(real t, real linear_predictor, real log_lambda, real log_k){
    real lambda = exp(log_lambda);
    real k = exp(log_k);
    return exp(linear_predictor)*log1p(pow(lambda*t, k));
  }
}

data{
  int<lower = 1> N;
  int<lower = 1> N_subjects;
  int<lower = 1> subject[N];
  int<lower = 0, upper = 1> event[N];
  vector[N] y;
  int<lower = 1> K;
  matrix[N, K] X;
  int<lower = 1> N_times;
  vector[N_times] times;
  int<lower = 0, upper = 1> status_brier[N, N_times];
}

parameters{
  vector[K] beta;
  real log_lambda;
  real log_k;
  real log_alpha;
  vector<lower = 0>[N_subjects] Z;
}

model{
  beta ~ normal(0, 1);
  log_lambda ~ normal(0, 1);
  log_k ~ normal(0, 1);
  log_alpha ~ normal(0, 1);
  Z ~ gamma(exp(log_alpha), exp(log_alpha));

  for(n in 1:N){
    real linear_predictor = dot_product(X[n], beta);
    real hazard = loglogistic_hazard(y[n], linear_predictor, log_lambda, log_k)*Z[subject[n]];
    real cumulative_hazard = loglogistic_cumulative_hazard(y[n], linear_predictor, log_lambda, log_k)*Z[subject[n]];
    target += event[n]*log(hazard) - cumulative_hazard;
  }
}

generated quantities{
  vector[N] log_lik;

  matrix[N, N_times] survival;
  vector[N_times] Brier_Score;
  vector[N_times] mean_survival;


  vector[N_times] Brier_alarm_B;
  vector[N_times] mean_survival_alarm_B;

  vector[N_times] Brier_alarm_not_B;
  vector[N_times] mean_survival_alarm_not_B;

  real variance_Z = 1/exp(log_alpha);
  real kendalls_tau_Z = exp(log_alpha)/(2 + exp(log_alpha));

  for(n in 1:N){
    real linear_predictor = dot_product(X[n], beta);
    real hazard = loglogistic_hazard(y[n], linear_predictor, log_lambda, log_k)*Z[subject[n]];
    real cumulative_hazard = loglogistic_cumulative_hazard(y[n], linear_predictor, log_lambda, log_k)*Z[subject[n]];
    log_lik[n] = event[n]*log(hazard) - cumulative_hazard;
  }

  for(t in 1:N_times){
    real total_survival = 0;
    real total_squared_error = 0;

    real total_survival_B = 0;
    real total_squared_error_B = 0;
    int count_B = 0;

    real total_survival_not_B = 0;
    real total_squared_error_not_B = 0;
    int count_not_B = 0;

    for(n in 1:N){
      real linear_predictor = dot_product(X[n], beta);
      real cumulative_hazard = loglogistic_cumulative_hazard(times[t], linear_predictor, log_lambda, log_k)*Z[subject[n]];
      real survival_probability = exp(-cumulative_hazard);
      real squared_error = square((1 - status_brier[n, t]) - survival_probability);

      survival[n, t] = survival_probability;

      total_survival += survival_probability;
      total_squared_error += squared_error;

      if(X[n,1] == 1){
        total_survival_B += survival_probability;
        total_squared_error_B += squared_error;
        count_B += 1;
      }
      if(X[n,1] == 0){
        total_survival_not_B += survival_probability;
        total_squared_error_not_B += squared_error;
        count_not_B += 1;
      }
    }

    mean_survival[t] = total_survival/N;
    Brier_Score[t] = total_squared_error/N;

    mean_survival_alarm_B[t] = count_B > 0 ? total_survival_B/count_B : 0;
    Brier_alarm_B[t] = count_B > 0 ? total_squared_error_B/count_B : 0;

    mean_survival_alarm_not_B[t] = count_not_B > 0 ? total_survival_not_B/count_not_B : 0;
    Brier_alarm_not_B[t] = count_not_B > 0 ? total_squared_error_not_B/count_not_B : 0;
  }
}
