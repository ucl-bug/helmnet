function a0_fit = fitPowerLawParamsMulti(a0, y, c0, f_ref, y_ref, plot_fit)
%FITPOWERLAWPARAMS Fit power law absorption parameters for highly absorbing media.
%
% DESCRIPTION:
%     fitPowerLawParamsMulti calculates the absorption parameters that
%     should be defined in the simulation functions given the desired
%     absorption behaviour (defined by a0 and y) at a given frequency
%     (f_ref). This takes into account the actual absorption behaviour 
%     exhibited by the fractional Laplacian wave equation (see Eq. 40 in
%     [1]), and the restriction that k-Wave only allows a single value for
%     the power law exponent medium.alpha_power (given here by y_ref).
%
%     This fitting is required when using large absorption values or high
%     frequencies, as the fractional Laplacian wave equation solved in
%     kspaceFirstOrderND and kspaceSecondOrder no longer encapsulates
%     absorption of the form a = a0*f^y (see Fig. 2 in [1]). It is also
%     required if using a spatially varying power law exponent y.
%
%     The returned values should be used to define medium.alpha_coeff
%     within the simulation functions, with medium.alpha_power = y_ref. The
%     absorption behaviour at the frequency f_ref will then match the
%     absorption given by the power law parameters a0 and y.
%
% USAGE:
%     a0_fit = fitPowerLawParamsMulti(a0, y, c0, f_ref, y_ref)
%     a0_fit = fitPowerLawParamsMulti(a0, y, c0, f_ref, y_ref, plot_fit)
%
% INPUTS:
%     a0          - matrix of desired power law absorption prefactors
%                   [dB/(MHz^y cm)] 
%     y           - matrix of desired power law exponents
%     c0          - matrix of medium sound speed [m/s]
%     f_ref       - reference frequency [Hz]
%     y_ref       - reference power law exponent
%
% OPTIONAL INPUTS
%     plot_fit    - Boolean controlling whether the final fit is
%                   displayed (default = false)
%
% OUTPUTS:
%     a0_fit      - power law absorption prefactor that should be used to
%                   define medium.alpha_coeff in the simulation functions
%
% ABOUT:
%     author      - Bradley E. Treeby
%     date        - 8th March 2018
%     last update - 26th April 2020
%
% REFERENCES:
%     [1] Treeby, B. E., & Cox, B. T. (2014). Modeling power law absorption
%     and dispersion in viscoelastic solids using a split-field and the
%     fractional Laplacian. The Journal of the Acoustical Society of
%     America, 136(4), 1499-1510.   
%
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2018-2019 Bradley Treeby

% This file is part of k-Wave. k-Wave is free software: you can
% redistribute it and/or modify it under the terms of the GNU Lesser
% General Public License as published by the Free Software Foundation,
% either version 3 of the License, or (at your option) any later version.
% 
% k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for
% more details. 
% 
% You should have received a copy of the GNU Lesser General Public License
% along with k-Wave. If not, see <http://www.gnu.org/licenses/>.

% check for plot input
if nargin < 6
    plot_fit = false;
end

% check inputs
validateattributes(a0,       {'numeric'}, {'real', 'nonnegative'},              mfilename, 'a0',       1);
validateattributes(y,        {'numeric'}, {'real', '>=', 0, '<=', 3},           mfilename, 'y',        2);
validateattributes(c0,       {'numeric'}, {'real', 'nonnegative'},              mfilename, 'c0',       3);
validateattributes(f_ref,    {'numeric'}, {'real', 'nonnegative', 'scalar'},    mfilename, 'f_ref',    4);
validateattributes(y_ref,    {'numeric'}, {'real', 'scalar', '>=', 0, '<=', 3}, mfilename, 'y_ref',    5);
validateattributes(plot_fit, {'logical'}, {'scalar'},                           mfilename, 'plot_fit', 6);

% make sure reference value isn't 1
if y_ref == 1
    error('Input for y_ref cannot be set to 1.');
end

% define frequency in rad/s
w = 2 * pi * f_ref;

% convert user defined a0 to Nepers/((rad/s)^y m)
a0_np = db2neper(a0, y);

% define desired absorption behaviour in Nepers/m
desired_absorption = a0_np .* w.^y;

% find the corresponding values of a0 that should be used in the
% fractional Laplacian wave equation to give the desired absorption
% behaviour taking into account second order effects (see Eq. 40 in [1])
a0_fit_np = desired_absorption ./ (  w.^y_ref + desired_absorption .* (y_ref + 1) .* c0 .* tan(pi .* y_ref ./ 2) .* w.^(y_ref - 1) );

% convert absorption prefactor back to dB/(MHz^y cm)
a0_fit = neper2db(a0_fit_np, y_ref);

% plot the final fit if desired
if plot_fit
    
    % create a small frequency range around the input frequency
    f_min = f_ref / 2;
    f_max = f_ref + f_min;
    f = f_min:(f_max - f_min)/1000:f_max;
    w = 2 * pi * f;
    
    % get suitable x-axis scale factor
    [~, scale, prefix] = scaleSI(f(end));
    
    % convert from Np/m to dB/cm
    conv_factor = (0.01 * 20 * log10(exp(1)));
    desired_absorption = desired_absorption .* conv_factor;
    
    % open figure
    figure;
    hold on;
    
    % get colors
    color_order = get(gca, 'ColorOrder');
    num_colors = size(color_order, 1);
    color_ind = 1;
    
    % get the unique values of the absorption coefficient
    [a0_uniq, ind_uniq] = unique(a0);
    
    % loop through the input values
    for plot_ind = 1:length(a0_uniq)
        
        % get the index of the value for comparison
        comp_ind = ind_uniq(plot_ind);
    
        % compute absorption behaviour
        absorption_fit = a0_fit_np(comp_ind) .* w.^y_ref ./ ( 1 - (y_ref + 1) .* a0_fit_np(comp_ind) .* c0(comp_ind) .* tan(pi .* y_ref ./ 2) .* w.^(y_ref - 1) );

        % convert from Np/m to dB/cm
        absorption_fit = absorption_fit .* conv_factor;

        % plot
        plot(f_ref .* scale, desired_absorption(comp_ind), '.', 'Color', color_order(color_ind, :));
        plot(f .* scale, a0(comp_ind) .* (f .* 1e-6) .^ y(comp_ind), '--', 'Color', color_order(color_ind, :));
        plot(f .* scale, absorption_fit, '-', 'Color', color_order(color_ind, :));

        % increment color index
        color_ind = color_ind + 1;
        if color_ind > num_colors
            color_ind = 1;
        end
        
    end
    
    % annotate plot
    set(gca, 'FontSize', 12);
    xlabel(['Frequency [' prefix 'Hz]']);
    ylabel('Absorption [dB/cm]');
    legend('Desired Absorption Value', 'Original Power Law', 'Fitted Power Law', 'Location', 'NorthWest');
    axis tight;
    
end