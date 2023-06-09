//auto-generated file from python - c++ friendly

//Specs:
//solver_dim = D3
//solver_type = INVERSE
//kernel_type = STANDARD
//min_itr = 1 (used in the database signature)
//max_itr = 60 (used in the database signature)
//max_rank = 8
//dx = 1.0
//alpha = 1.0
//beta = 6.0
//decomp_method = SYM_CP_3D

//------------------itr = 60------------------
static const int INVERSE_Itr_60_Filter_Size = 119;
static const int INVERSE_Itr_60_Half_Filter_Size = 59;
static const float4 INVERSE_Itr_60_R14_Filters[] = {
//Rank 1  //Rank 2   //Rank 3   //Rank 4
float4(-2.10e-47, -7.25e-48, -7.07e-47, 2.75e-47),
float4(-2.68e-45, -3.39e-47, 2.57e-45, 2.69e-45),
float4(-2.79e-43, -7.08e-45, -2.88e-43, 7.96e-44),
float4(-1.74e-41, 9.92e-43, 1.65e-41, 2.56e-42),
float4(-9.33e-40, 2.35e-41, -3.34e-40, -2.17e-40),
float4(-3.67e-38, 1.59e-39, 3.49e-38, -1.25e-38),
float4(-1.30e-36, 4.37e-39, -5.57e-38, -7.74e-37),
float4(-3.68e-35, -9.87e-38, 3.47e-35, -2.52e-35),
float4(-9.59e-34, -4.16e-35, 1.37e-34, -8.34e-34),
float4(-2.09e-32, -1.25e-33, 1.95e-32, -1.97e-32),
float4(-4.25e-31, -4.19e-32, 1.13e-31, -4.63e-31),
float4(-7.43e-30, -8.92e-31, 6.84e-30, -8.57e-30),
float4(-1.23e-28, -1.91e-29, 4.36e-29, -1.56e-28),
float4(-1.78e-27, -3.18e-28, 1.61e-27, -2.35e-27),
float4(-2.44e-26, -5.15e-27, 1.03e-26, -3.46e-26),
float4(-2.98e-25, -7.01e-26, 2.67e-25, -4.38e-25),
float4(-3.47e-24, -9.18e-25, 1.65e-24, -5.37e-24),
float4(-3.65e-23, -1.05e-23, 3.22e-23, -5.80e-23),
float4(-3.65e-22, -1.15e-22, 1.90e-22, -6.04e-22),
float4(-3.33e-21, -1.12e-21, 2.90e-21, -5.64e-21),
float4(-2.90e-20, -1.04e-20, 1.61e-20, -5.07e-20),
float4(-2.32e-19, -8.83e-20, 1.99e-19, -4.13e-19),
float4(-1.77e-18, -7.12e-19, 1.04e-18, -3.24e-18),
float4(-1.25e-17, -5.27e-18, 1.06e-17, -2.33e-17),
float4(-8.43e-17, -3.72e-17, 5.15e-17, -1.61e-16),
float4(-5.29e-16, -2.43e-16, 4.44e-16, -1.02e-15),
float4(-3.17e-15, -1.51e-15, 2.01e-15, -6.25e-15),
float4(-1.78e-14, -8.78e-15, 1.48e-14, -3.54e-14),
float4(-9.53e-14, -4.85e-14, 6.22e-14, -1.93e-13),
float4(-4.79e-13, -2.51e-13, 3.94e-13, -9.79e-13),
float4(-2.30e-12, -1.24e-12, 1.54e-12, -4.77e-12),
float4(-1.04e-11, -5.74e-12, 8.49e-12, -2.18e-11),
float4(-4.51e-11, -2.54e-11, 3.08e-11, -9.52e-11),
float4(-1.84e-10, -1.06e-10, 1.49e-10, -3.92e-10),
float4(-7.19e-10, -4.21e-10, 5.00e-10, -1.54e-09),
float4(-2.65e-09, -1.58e-09, 2.13e-09, -5.74e-09),
float4(-9.40e-09, -5.68e-09, 6.65e-09, -2.05e-08),
float4(-3.15e-08, -1.93e-08, 2.51e-08, -6.90e-08),
float4(-1.01e-07, -6.28e-08, 7.25e-08, -2.23e-07),
float4(-3.09e-07, -1.93e-07, 2.44e-07, -6.82e-07),
float4(-9.02e-07, -5.69e-07, 6.54e-07, -2.00e-06),
float4(-2.51e-06, -1.59e-06, 1.97e-06, -5.58e-06),
float4(-6.69e-06, -4.26e-06, 4.89e-06, -1.49e-05),
float4(-1.70e-05, -1.09e-05, 1.32e-05, -3.79e-05),
float4(-4.15e-05, -2.64e-05, 3.05e-05, -9.23e-05),
float4(-9.66e-05, -6.14e-05, 7.47e-05, -2.14e-04),
float4(-2.16e-04, -1.36e-04, 1.60e-04, -4.78e-04),
float4(-4.63e-04, -2.89e-04, 3.55e-04, -1.02e-03),
float4(-9.56e-04, -5.86e-04, 7.07e-04, -2.08e-03),
float4(-1.89e-03, -1.14e-03, 1.43e-03, -4.08e-03),
float4(-3.62e-03, -2.10e-03, 2.66e-03, -7.67e-03),
float4(-6.67e-03, -3.73e-03, 4.97e-03, -1.38e-02),
float4(-1.19e-02, -6.30e-03, 8.64e-03, -2.40e-02),
float4(-2.07e-02, -1.02e-02, 1.49e-02, -4.00e-02),
float4(-3.51e-02, -1.56e-02, 2.43e-02, -6.38e-02),
float4(-5.84e-02, -2.26e-02, 3.86e-02, -9.72e-02),
float4(-9.66e-02, -3.10e-02, 5.66e-02, -1.39e-01),
float4(-1.61e-01, -4.04e-02, 7.06e-02, -1.79e-01),
float4(-2.82e-01, -5.50e-02, 2.12e-02, -1.71e-01),
float4(-5.48e-01, -1.09e-01, -4.34e-01, 5.58e-02),
float4(-2.82e-01, -5.50e-02, 2.12e-02, -1.71e-01),
float4(-1.61e-01, -4.04e-02, 7.06e-02, -1.79e-01),
float4(-9.66e-02, -3.10e-02, 5.66e-02, -1.39e-01),
float4(-5.84e-02, -2.26e-02, 3.86e-02, -9.72e-02),
float4(-3.51e-02, -1.56e-02, 2.43e-02, -6.38e-02),
float4(-2.07e-02, -1.02e-02, 1.49e-02, -4.00e-02),
float4(-1.19e-02, -6.30e-03, 8.64e-03, -2.40e-02),
float4(-6.67e-03, -3.73e-03, 4.97e-03, -1.38e-02),
float4(-3.62e-03, -2.10e-03, 2.66e-03, -7.67e-03),
float4(-1.89e-03, -1.14e-03, 1.43e-03, -4.08e-03),
float4(-9.56e-04, -5.86e-04, 7.07e-04, -2.08e-03),
float4(-4.63e-04, -2.89e-04, 3.55e-04, -1.02e-03),
float4(-2.16e-04, -1.36e-04, 1.60e-04, -4.78e-04),
float4(-9.66e-05, -6.14e-05, 7.47e-05, -2.14e-04),
float4(-4.15e-05, -2.64e-05, 3.05e-05, -9.23e-05),
float4(-1.70e-05, -1.09e-05, 1.32e-05, -3.79e-05),
float4(-6.69e-06, -4.26e-06, 4.89e-06, -1.49e-05),
float4(-2.51e-06, -1.59e-06, 1.97e-06, -5.58e-06),
float4(-9.02e-07, -5.69e-07, 6.54e-07, -2.00e-06),
float4(-3.09e-07, -1.93e-07, 2.44e-07, -6.82e-07),
float4(-1.01e-07, -6.28e-08, 7.25e-08, -2.23e-07),
float4(-3.15e-08, -1.93e-08, 2.51e-08, -6.90e-08),
float4(-9.40e-09, -5.68e-09, 6.65e-09, -2.05e-08),
float4(-2.65e-09, -1.58e-09, 2.13e-09, -5.74e-09),
float4(-7.19e-10, -4.21e-10, 5.00e-10, -1.54e-09),
float4(-1.84e-10, -1.06e-10, 1.49e-10, -3.92e-10),
float4(-4.51e-11, -2.54e-11, 3.08e-11, -9.52e-11),
float4(-1.04e-11, -5.74e-12, 8.49e-12, -2.18e-11),
float4(-2.30e-12, -1.24e-12, 1.54e-12, -4.77e-12),
float4(-4.79e-13, -2.51e-13, 3.94e-13, -9.79e-13),
float4(-9.53e-14, -4.85e-14, 6.22e-14, -1.93e-13),
float4(-1.78e-14, -8.78e-15, 1.48e-14, -3.54e-14),
float4(-3.17e-15, -1.51e-15, 2.01e-15, -6.25e-15),
float4(-5.29e-16, -2.43e-16, 4.44e-16, -1.02e-15),
float4(-8.43e-17, -3.72e-17, 5.15e-17, -1.61e-16),
float4(-1.25e-17, -5.27e-18, 1.06e-17, -2.33e-17),
float4(-1.77e-18, -7.12e-19, 1.04e-18, -3.24e-18),
float4(-2.32e-19, -8.83e-20, 1.99e-19, -4.13e-19),
float4(-2.90e-20, -1.04e-20, 1.61e-20, -5.07e-20),
float4(-3.33e-21, -1.12e-21, 2.90e-21, -5.64e-21),
float4(-3.65e-22, -1.15e-22, 1.90e-22, -6.04e-22),
float4(-3.65e-23, -1.05e-23, 3.22e-23, -5.80e-23),
float4(-3.47e-24, -9.18e-25, 1.65e-24, -5.37e-24),
float4(-2.98e-25, -7.01e-26, 2.67e-25, -4.38e-25),
float4(-2.44e-26, -5.15e-27, 1.03e-26, -3.46e-26),
float4(-1.78e-27, -3.18e-28, 1.61e-27, -2.35e-27),
float4(-1.23e-28, -1.91e-29, 4.36e-29, -1.56e-28),
float4(-7.43e-30, -8.92e-31, 6.84e-30, -8.57e-30),
float4(-4.25e-31, -4.19e-32, 1.13e-31, -4.63e-31),
float4(-2.09e-32, -1.25e-33, 1.95e-32, -1.97e-32),
float4(-9.59e-34, -4.16e-35, 1.37e-34, -8.34e-34),
float4(-3.68e-35, -9.87e-38, 3.47e-35, -2.52e-35),
float4(-1.30e-36, 4.37e-39, -5.57e-38, -7.74e-37),
float4(-3.67e-38, 1.59e-39, 3.49e-38, -1.25e-38),
float4(-9.33e-40, 2.35e-41, -3.34e-40, -2.17e-40),
float4(-1.74e-41, 9.92e-43, 1.65e-41, 2.56e-42),
float4(-2.79e-43, -7.08e-45, -2.88e-43, 7.96e-44),
float4(-2.68e-45, -3.39e-47, 2.57e-45, 2.69e-45),
float4(-2.10e-47, -7.25e-48, -7.07e-47, 2.75e-47)
};
static const float4 INVERSE_Itr_60_R58_Filters[] = {
//Rank 5  //Rank 6   //Rank 7   //Rank 8
float4(2.45e-47, -1.77e-47, -3.49e-47, 4.33e-47),
float4(-2.08e-45, -5.14e-45, 6.42e-45, -9.61e-46),
float4(1.74e-43, -2.23e-44, -3.14e-43, -1.18e-42),
float4(-5.85e-42, -1.65e-41, 4.73e-41, -2.72e-41),
float4(4.57e-40, 2.59e-40, -3.16e-40, -3.65e-39),
float4(-5.11e-39, -1.53e-38, 9.24e-38, -5.08e-38),
float4(6.04e-37, 6.34e-37, -2.43e-38, -4.07e-36),
float4(-2.43e-37, -3.20e-36, 8.03e-35, -3.52e-35),
float4(4.50e-34, 5.96e-34, 3.73e-35, -2.27e-33),
float4(1.98e-33, 2.85e-33, 3.75e-32, -1.07e-32),
float4(2.07e-31, 3.03e-31, -1.90e-32, -7.23e-31),
float4(1.32e-30, 2.21e-30, 1.04e-29, -8.31e-31),
float4(6.20e-29, 9.53e-29, -2.52e-29, -1.39e-28),
float4(4.37e-28, 7.41e-28, 1.76e-27, 4.27e-28),
float4(1.28e-26, 2.01e-26, -9.89e-27, -1.59e-26),
float4(9.10e-26, 1.52e-25, 1.78e-25, 1.63e-25),
float4(1.90e-24, 2.97e-24, -2.17e-24, -8.15e-25),
float4(1.30e-23, 2.13e-23, 8.18e-24, 2.95e-23),
float4(2.06e-22, 3.22e-22, -3.11e-22, 4.49e-23),
float4(1.33e-21, 2.14e-21, -4.29e-22, 3.44e-21),
float4(1.68e-20, 2.61e-20, -3.11e-20, 1.25e-20),
float4(1.02e-19, 1.60e-19, -1.06e-19, 2.85e-19),
float4(1.06e-18, 1.62e-18, -2.28e-18, 1.24e-18),
float4(5.92e-18, 9.15e-18, -9.58e-18, 1.75e-17),
float4(5.15e-17, 7.81e-17, -1.26e-16, 7.89e-17),
float4(2.67e-16, 4.06e-16, -5.56e-16, 8.16e-16),
float4(1.98e-15, 2.96e-15, -5.37e-15, 3.62e-15),
float4(9.46e-15, 1.41e-14, -2.33e-14, 2.97e-14),
float4(6.05e-14, 8.96e-14, -1.78e-13, 1.26e-13),
float4(2.66e-13, 3.93e-13, -7.41e-13, 8.51e-13),
float4(1.48e-12, 2.17e-12, -4.68e-12, 3.40e-12),
float4(6.01e-12, 8.75e-12, -1.83e-11, 1.95e-11),
float4(2.94e-11, 4.27e-11, -9.83e-11, 7.25e-11),
float4(1.10e-10, 1.58e-10, -3.59e-10, 3.58e-10),
float4(4.75e-10, 6.82e-10, -1.66e-09, 1.24e-09),
float4(1.62e-09, 2.32e-09, -5.63e-09, 5.34e-09),
float4(6.26e-09, 8.92e-09, -2.27e-08, 1.71e-08),
float4(1.97e-08, 2.79e-08, -7.12e-08, 6.48e-08),
float4(6.78e-08, 9.60e-08, -2.54e-07, 1.92e-07),
float4(1.96e-07, 2.76e-07, -7.32e-07, 6.46e-07),
float4(6.07e-07, 8.54e-07, -2.32e-06, 1.76e-06),
float4(1.62e-06, 2.27e-06, -6.14e-06, 5.29e-06),
float4(4.50e-06, 6.31e-06, -1.74e-05, 1.33e-05),
float4(1.10e-05, 1.54e-05, -4.23e-05, 3.57e-05),
float4(2.78e-05, 3.89e-05, -1.07e-04, 8.28e-05),
float4(6.27e-05, 8.77e-05, -2.40e-04, 2.00e-04),
float4(1.43e-04, 2.01e-04, -5.48e-04, 4.26e-04),
float4(2.98e-04, 4.19e-04, -1.12e-03, 9.28e-04),
float4(6.22e-04, 8.76e-04, -2.31e-03, 1.81e-03),
float4(1.20e-03, 1.69e-03, -4.30e-03, 3.56e-03),
float4(2.28e-03, 3.24e-03, -7.96e-03, 6.32e-03),
float4(4.06e-03, 5.79e-03, -1.34e-02, 1.11e-02),
float4(7.11e-03, 1.02e-02, -2.21e-02, 1.76e-02),
float4(1.17e-02, 1.68e-02, -3.31e-02, 2.68e-02),
float4(1.89e-02, 2.70e-02, -4.71e-02, 3.51e-02),
float4(2.90e-02, 4.01e-02, -5.84e-02, 3.87e-02),
float4(4.33e-02, 5.48e-02, -6.24e-02, 2.08e-02),
float4(6.32e-02, 5.85e-02, -4.53e-02, -3.90e-02),
float4(1.02e-01, 1.21e-02, -1.51e-02, -1.47e-01),
float4(2.59e-01, -2.10e-01, -1.24e-01, -4.67e-02),
float4(1.02e-01, 1.21e-02, -1.51e-02, -1.47e-01),
float4(6.32e-02, 5.85e-02, -4.53e-02, -3.90e-02),
float4(4.33e-02, 5.48e-02, -6.24e-02, 2.08e-02),
float4(2.90e-02, 4.01e-02, -5.84e-02, 3.87e-02),
float4(1.89e-02, 2.70e-02, -4.71e-02, 3.51e-02),
float4(1.17e-02, 1.68e-02, -3.31e-02, 2.68e-02),
float4(7.11e-03, 1.02e-02, -2.21e-02, 1.76e-02),
float4(4.06e-03, 5.79e-03, -1.34e-02, 1.11e-02),
float4(2.28e-03, 3.24e-03, -7.96e-03, 6.32e-03),
float4(1.20e-03, 1.69e-03, -4.30e-03, 3.56e-03),
float4(6.22e-04, 8.76e-04, -2.31e-03, 1.81e-03),
float4(2.98e-04, 4.19e-04, -1.12e-03, 9.28e-04),
float4(1.43e-04, 2.01e-04, -5.48e-04, 4.26e-04),
float4(6.27e-05, 8.77e-05, -2.40e-04, 2.00e-04),
float4(2.78e-05, 3.89e-05, -1.07e-04, 8.28e-05),
float4(1.10e-05, 1.54e-05, -4.23e-05, 3.57e-05),
float4(4.50e-06, 6.31e-06, -1.74e-05, 1.33e-05),
float4(1.62e-06, 2.27e-06, -6.14e-06, 5.29e-06),
float4(6.07e-07, 8.54e-07, -2.32e-06, 1.76e-06),
float4(1.96e-07, 2.76e-07, -7.32e-07, 6.46e-07),
float4(6.78e-08, 9.60e-08, -2.54e-07, 1.92e-07),
float4(1.97e-08, 2.79e-08, -7.12e-08, 6.48e-08),
float4(6.26e-09, 8.92e-09, -2.27e-08, 1.71e-08),
float4(1.62e-09, 2.32e-09, -5.63e-09, 5.34e-09),
float4(4.75e-10, 6.82e-10, -1.66e-09, 1.24e-09),
float4(1.10e-10, 1.58e-10, -3.59e-10, 3.58e-10),
float4(2.94e-11, 4.27e-11, -9.83e-11, 7.25e-11),
float4(6.01e-12, 8.75e-12, -1.83e-11, 1.95e-11),
float4(1.48e-12, 2.17e-12, -4.68e-12, 3.40e-12),
float4(2.66e-13, 3.93e-13, -7.41e-13, 8.51e-13),
float4(6.05e-14, 8.96e-14, -1.78e-13, 1.26e-13),
float4(9.46e-15, 1.41e-14, -2.33e-14, 2.97e-14),
float4(1.98e-15, 2.96e-15, -5.37e-15, 3.62e-15),
float4(2.67e-16, 4.06e-16, -5.56e-16, 8.16e-16),
float4(5.15e-17, 7.81e-17, -1.26e-16, 7.89e-17),
float4(5.92e-18, 9.15e-18, -9.58e-18, 1.75e-17),
float4(1.06e-18, 1.62e-18, -2.28e-18, 1.24e-18),
float4(1.02e-19, 1.60e-19, -1.06e-19, 2.85e-19),
float4(1.68e-20, 2.61e-20, -3.11e-20, 1.25e-20),
float4(1.33e-21, 2.14e-21, -4.29e-22, 3.44e-21),
float4(2.06e-22, 3.22e-22, -3.11e-22, 4.49e-23),
float4(1.30e-23, 2.13e-23, 8.18e-24, 2.95e-23),
float4(1.90e-24, 2.97e-24, -2.17e-24, -8.15e-25),
float4(9.10e-26, 1.52e-25, 1.78e-25, 1.63e-25),
float4(1.28e-26, 2.01e-26, -9.89e-27, -1.59e-26),
float4(4.37e-28, 7.41e-28, 1.76e-27, 4.27e-28),
float4(6.20e-29, 9.53e-29, -2.52e-29, -1.39e-28),
float4(1.32e-30, 2.21e-30, 1.04e-29, -8.31e-31),
float4(2.07e-31, 3.03e-31, -1.90e-32, -7.23e-31),
float4(1.98e-33, 2.85e-33, 3.75e-32, -1.07e-32),
float4(4.50e-34, 5.96e-34, 3.73e-35, -2.27e-33),
float4(-2.43e-37, -3.20e-36, 8.03e-35, -3.52e-35),
float4(6.04e-37, 6.34e-37, -2.43e-38, -4.07e-36),
float4(-5.11e-39, -1.53e-38, 9.24e-38, -5.08e-38),
float4(4.57e-40, 2.59e-40, -3.16e-40, -3.65e-39),
float4(-5.85e-42, -1.65e-41, 4.73e-41, -2.72e-41),
float4(1.74e-43, -2.23e-44, -3.14e-43, -1.18e-42),
float4(-2.08e-45, -5.14e-45, 6.42e-45, -9.61e-46),
float4(2.45e-47, -1.77e-47, -3.49e-47, 4.33e-47)
};