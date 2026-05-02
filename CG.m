%% === Parameter and model setup ===
clc
clear all
close all

%% --- Geometry ---
length_real = 0.410;     % [m]
width_real  = 0.0939;    % [m]
height_real = 0.149;     % [m]

% Ellipsoid semi-axes
a = length_real/2;       % [m]
b = width_real/2;        % [m]
c = height_real/2;       % [m]

%% --- Fluid and mass ---
rho = 1000;              % [kg/m^3]
g   = 9.81;              % [m/s^2]

m = 3;                   % [kg]
W = m*g;                 % [N]

% Ellipsoid volume and geometric buoyancy check
V_body = (4/3)*pi*a*b*c;     % [m^3]
B_geo  = rho*g*V_body;       % [N]

% Use neutral buoyancy in simulation
B = W;

%% --- Body projected areas ---
A_x = pi*b*c;            % [m^2] projected area normal to x, surge
A_y = pi*a*c;            % [m^2] projected area normal to y, sway
A_z = pi*a*b;            % [m^2] projected area normal to z, heave

% Ellipsoid surface area approximation, Knud Thomsen formula
p_surf = 1.6075;
A_Body = 4*pi*(((a*b)^p_surf + (a*c)^p_surf + (b*c)^p_surf)/3)^(1/p_surf);

%% --- Mass moment of inertia ---
% CAD / measured values retained
Ixx = 0.0042;            % [kg*m^2]
Iyy = 0.0274;            % [kg*m^2]
Izz = 0.0274;            % [kg*m^2]

% Ellipsoid theoretical check values
Ixx_ellip = (1/5)*m*(b^2 + c^2);
Iyy_ellip = (1/5)*m*(a^2 + c^2);
Izz_ellip = (1/5)*m*(a^2 + b^2);

%% --- Nonlinear translational drag ---
% Surge direction is streamlined, cross-flow directions use larger Cd.
Cd_x = 0.04;
Cd_y = 0.40;
Cd_z = 0.40;

Xuu = -0.5*rho*Cd_x*A_x;     % [kg/m], X_{u|u|}
Yvv = -0.5*rho*Cd_y*A_y;     % [kg/m], Y_{v|v|}
Zww = -0.5*rho*Cd_z*A_z;     % [kg/m], Z_{w|w|}

%% --- Added mass: triaxial ellipsoid approximation ---
% General triaxial ellipsoid shape coefficients
deltaFun = @(lam) sqrt((a^2 + lam).*(b^2 + lam).*(c^2 + lam));

alpha0 = a*b*c * integral(@(lam) ...
    1./((a^2 + lam).*deltaFun(lam)), 0, Inf);

beta0 = a*b*c * integral(@(lam) ...
    1./((b^2 + lam).*deltaFun(lam)), 0, Inf);

gamma0 = a*b*c * integral(@(lam) ...
    1./((c^2 + lam).*deltaFun(lam)), 0, Inf);

% Added-mass derivatives
Xud = -rho*V_body*alpha0/(2 - alpha0);     % [kg], X_udot
Yvd = -rho*V_body*beta0 /(2 - beta0);      % [kg], Y_vdot
Zwd = -rho*V_body*gamma0/(2 - gamma0);     % [kg], Z_wdot

% Added-inertia derivatives
Kpd = -(1/5)*rho*V_body * ...
    ((b^2 - c^2)^2*(gamma0 - beta0)) / ...
    (2*(b^2 - c^2) + (b^2 + c^2)*(beta0 - gamma0));

Mqd = -(1/5)*rho*V_body * ...
    ((c^2 - a^2)^2*(alpha0 - gamma0)) / ...
    (2*(c^2 - a^2) + (c^2 + a^2)*(gamma0 - alpha0));

Nrd = -(1/5)*rho*V_body * ...
    ((a^2 - b^2)^2*(beta0 - alpha0)) / ...
    (2*(a^2 - b^2) + (a^2 + b^2)*(alpha0 - beta0));

%% --- Cross added-mass terms ---
% For a symmetric ellipsoid with body-fixed principal axes,
% these cross terms should be zero unless experimentally identified.
Yrd = 0;
Zqd = 0;

%% --- Quadratic rotational damping ---
% Body-only estimate is much smaller than the empirical value.
% Therefore empirical values are retained here for simulation robustness.

Kpp = 0;

use_empirical_rot_damping = true;

% Body-only rough estimates, not used unless the flag is false
Mqq_body_est = -(4/15)*rho*Cd_z*b*a^4;
Nrr_body_est = -(4/15)*rho*Cd_y*c*a^4;

if use_empirical_rot_damping
    Mqq = -0.3315;       % [kg*m^2], empirical / tuned
    Nrr = -0.3315;       % [kg*m^2], empirical / tuned
else
    Mqq = Mqq_body_est;
    Nrr = Nrr_body_est;
end

%% --- Slip damping ---
% Empirical terms retained
Mww = -5.0;
Nvv = -5.0;

%% --- Centers ---
% Body-fixed frame origin is assumed at geometric center.
xb = 0.0;
zb = 0.0;

xG = 0.0;

% NED convention: positive z is downward.
% zG > 0 means CG is below CB, which gives restoring stability.
zG = 0.004535;

%% --- Fin action points ---
% Dorsal fin
xd = 0.139220;
zd = -0.048232;

% Pectoral fins
xp = 0.130909;

% Pectoral fin is below center.
% In NED convention, below center is positive z.
zp = +0.000417;

yp = 0.045097;

%% --- Fin areas ---
% These values are retained as selected/CAD-based fin areas.
% A_pectoral is assumed to be total area of both pectoral fins.
A_pectoral = 0.013092;      % [m^2]
A_dorsal   = 0.006271;      % [m^2]

% If the model requires one pectoral fin area, use:
A_pectoral_each = A_pectoral/2;

%% --- Additional parameter ---
radius = 15;

%% --- System matrix setup ---
X1 = zeros(9,9);

X1(1,1) = m - Xud;
X1(1,5) = m*zG;

X1(2,2) = m - Yvd;
X1(2,4) = -m*zG;
X1(2,6) = m*xG - Yrd;

X1(3,3) = m - Zwd;
X1(3,5) = -m*xG - Zqd;

X1(4,2) = -m*zG;
X1(4,4) = Ixx - Kpd;

X1(5,1) = m*zG;
X1(5,3) = -m*xG - Zqd;
X1(5,5) = Iyy - Mqd;

X1(6,2) = m*xG - Yrd;
X1(6,6) = Izz - Nrd;

X1(7,7) = 1;
X1(8,8) = 1;
X1(9,9) = 1;

%% --- State-space matrices ---
A1 = zeros(9,9);
B1 = X1 \ eye(9);
C1 = eye(9,9);
D1 = zeros(9,9);

%% --- Auxiliary terms ---
% Do not use xG + Zqd because xG and Zqd have different physical dimensions.
xG_plus_zq = xG;

W_zG_zb = W*(zG - zb);
invDen  = 1/(Iyy - Mqd);

%% --- Display check ---
fprintf('\n=== Geometry check ===\n');
fprintf('a = %.6f m, b = %.6f m, c = %.6f m\n', a, b, c);
fprintf('V_body = %.9f m^3\n', V_body);
fprintf('rho*V_body = %.6f kg\n', rho*V_body);
fprintf('W = %.6f N, B_geo = %.6f N, B_used = %.6f N\n', W, B_geo, B);

fprintf('\n=== Area check ===\n');
fprintf('A_x = %.8f m^2\n', A_x);
fprintf('A_y = %.8f m^2\n', A_y);
fprintf('A_z = %.8f m^2\n', A_z);
fprintf('A_Body(surface) = %.8f m^2\n', A_Body);
fprintf('A_pectoral(total) = %.8f m^2\n', A_pectoral);
fprintf('A_dorsal = %.8f m^2\n', A_dorsal);

fprintf('\n=== Drag coefficients ===\n');
fprintf('Xuu = %.6f kg/m\n', Xuu);
fprintf('Yvv = %.6f kg/m\n', Yvv);
fprintf('Zww = %.6f kg/m\n', Zww);
fprintf('|Yvv/Xuu| = %.3f\n', abs(Yvv/Xuu));
fprintf('|Zww/Xuu| = %.3f\n', abs(Zww/Xuu));

fprintf('\n=== Added mass ===\n');
fprintf('alpha0 = %.6f, beta0 = %.6f, gamma0 = %.6f\n', alpha0, beta0, gamma0);
fprintf('Xud = %.6f kg\n', Xud);
fprintf('Yvd = %.6f kg\n', Yvd);
fprintf('Zwd = %.6f kg\n', Zwd);
fprintf('Kpd = %.8f kg*m^2\n', Kpd);
fprintf('Mqd = %.8f kg*m^2\n', Mqd);
fprintf('Nrd = %.8f kg*m^2\n', Nrd);

fprintf('\n=== Inertia check ===\n');
fprintf('Ixx used = %.6f, ellipsoid = %.6f kg*m^2\n', Ixx, Ixx_ellip);
fprintf('Iyy used = %.6f, ellipsoid = %.6f kg*m^2\n', Iyy, Iyy_ellip);
fprintf('Izz used = %.6f, ellipsoid = %.6f kg*m^2\n', Izz, Izz_ellip);

fprintf('\n=== Matrix check ===\n');
fprintf('cond(X1) = %.6e\n', cond(X1));
