function [medium, skull_mask, head_mask, air_mask] = skull2medium(ct_img, ct_calibration, f_ref, varargin)
%SKULL2MEDIUM Convert skull CT scan to medium structure.
%
% DESCRIPTION:
%     skull2medium converts a CT image of a head and skull to the acoustic
%     and thermal properties used by the k-Wave simulation functions. The
%     image is divided into four regions by thresholding - background, soft
%     tissue, skull, and air. Constant properties are assigned within the
%     background, soft tissue, and air regions. Within the bone, the
%     hounsfield units are directly mapped to the sound speed and density,
%     while constant properties are used for the attenuation and thermal
%     properties.
%
%     The skull, head, and air masks are calculated by thresholding ct_img.
%     The threshold values can be modified using the optional input
%     parameters 'SkullThreshold' and 'HeadThreshold', where
%
%        skull_mask = (ct_img >= skull_threshold)
%        head_mask  = (ct_img >= head_threshold)
%        air_mask   = (ct_img <  head_threshold)
%
%     After thresholding, a series of morphological operations are
%     performed to improved the segmetation. For the skull_mask, only the
%     largest N connected components are retained in the masks. The number
%     of connected components can be modified using the optional input
%     parameter 'UseLargestCC'. Small holes are then filled, and the
%     air_mask is subtracted. For the head_mask, the complete mask is
%     filled (thus the head_mask contains the air_mask and skull_mask). For
%     the air_mask, only the air within the head_mask is retained, small
%     clusters are removed, and small holes are filled.
%
%     Within the skull mask, the CT Hounsfield units (HU) are converted to
%     mass density using the user defined calibration curve. This curve is
%     defined as a piecewise linear relationship between HU and density in
%     kg/m^3. This can be obtained by using a CT scan of a density phantom,
%     e.g., the CIRS Model 062M electron density phantom. If a conversion
%     is not available, ct_calibration can be set to [], in which case the
%     conversion is performed by hounsfield2density.
%   
%     The skull sound speed is calculated from the density assuming a
%     linear relationship, where c = rho * rho2c_slope + rho2c_intcp. The
%     values for the slope and intercept can be modified using the optional
%     input parameter 'ConversionSettings'.
%
%     A homogeneous value for the power law attenuation coefficient and
%     exponent within the skull is assumed. These values can be modified
%     using the optional input parameter 'ConversionSettings'.
%
%     Values for the properties outside the skull mask are replaced with
%     either tissue, air, or background values. The reference sound speed
%     and diffusion coefficient are set to the background values. These can
%     be modified using the optional input parameter 'ConversionSettings'.
%
%     To account for the high attenuation within the skull, and a differing
%     power law exponent between the skull and background, the absorption
%     coefficient values are modified using the function
%     fitPowerLawParamsMulti along with the reference frequency f_ref. The
%     reference power law exponent is always chosen to be y = 2.
% 
%     Impedance changes between air and tissue can cause stability problems
%     for k-Wave, thus the air density is artificially increased by a
%     factor of 100. This can be modified using the optional
%     input parameter 'ConversionSettings'. The air values can also be
%     excluded from the conversion by setting the optional input parameter
%     'IncludeAir' to false.
%
%     Note, the CT image data must be in conventional HU units, where
%         HU = 1000 * (mu - mu_water) / (mu_water - mu_air)
%     Sometimes the image data can be defined in scaled HU units, where
%         HU_sc = 1000 * mu / mu_water,
%     In this case, HU = HU_sc - 1000, where mu_air is approximately 0. 
%
% USAGE:
%     medium = skull2medium(ct_img, [], f_ref)
%     medium = skull2medium(ct_img, ct_calibration, f_ref)
%     [medium, skull_mask, head_mask, air_mask] = skull2medium(ct_img, ct_calibration, f_ref)
%     [medium, skull_mask, head_mask, air_mask] = skull2medium(ct_img, ct_calibration, f_ref, ...)
%
% INPUTS:
%     ct_img        - CT image of the head and skull [HU]. 
%
%     ct_calibration
%                   - 2xN array giving a piecewise linear relationship
%                     between HU and mass density obtained using a
%                     stoichoimetric calibration of the CT scanner. The
%                     first row is HU and the second row is mass density in
%                     [kg/m^3]. If set to [], the conversion is calculated
%                     using the k-Wave function hounsfield2density.
%
%     f_ref         - Ultrasound frequency used to calculate the absorption
%                     coefficient [Hz]. This can be defined as [] if only
%                     returning the thermal properties.
%
% OPTIONAL INPUTS:
%     Optional 'string', value pairs that may be used to modify the default
%     computational  
%
%     'ConversionSettings'
%                   - Structure defining the literals used for the
%                     conversion. Not all parameters need to be defined.
%
%                     settings.skull_rho2c_slope            [m^4/kg.s]
%                     settings.skull_rho2c_intcp            [m/s]
%                     settings.skull_alpha_coeff            [dB/(MHz^y cm)]
%                     settings.skull_alpha_power            [-]
%                     settings.skull_BonA                   [-]
%                     settings.skull_thermal_conductivity   [W/(m.K)]
%                     settings.skull_specific_heat          [J/(kg.K)]
%
%                     settings.bg_sound_speed               [m/s]
%                     settings.bg_density                   [kg/m^3]
%                     settings.bg_alpha_coeff               [dB/(MHz^y cm)]
%                     settings.bg_alpha_power               [-]
%                     settings.bg_BonA                      [-]
%                     settings.bg_thermal_conductivity      [W/(m.K)]
%                     settings.bg_specific_heat             [J/(kg.K)]
%
%                     settings.tissue_sound_speed           [m/s]
%                     settings.tissue_density               [kg/m^3]
%                     settings.tissue_alpha_coeff           [dB/(MHz^y cm)]
%                     settings.tissue_alpha_power           [-]
%                     settings.tissue_BonA                  [-]
%                     settings.tissue_thermal_conductivity  [W/(m.K)]
%                     settings.tissue_specific_heat         [J/(kg.K)]
%
%                     settings.air_sound_speed              [m/s]
%                     settings.air_density                  [kg/m^3]
%                     settings.air_alpha_coeff              [dB/(MHz^y cm)]
%                     settings.air_alpha_power              [-]
%                     settings.air_BonA                     [-]
%                     settings.air_thermal_conductivity     [W/(m.K)]
%                     settings.air_specific_heat            [J/(kg.K)]
%
%     'HeadThreshold'
%                   - Threshold parameter (given in Hounsfield units) used
%                     to automatically define a head mask (default = -250).
%
%     'IncludeAir'  - Boolean controlling whether the regions of air are
%                     assigned the properties of air (true) or tissue
%                     (false). Default = true.
%
%     'Plot'        - Boolean controlling whether the calculated medium
%                     properties of the central slice are plotted (default
%                     = false).
%
%     'ReturnValues'
%                   - String controlling which properties are returned. Can
%                     be set to return subsets of the medium properties:
%                         'all'               - all properties
%                         'acoustic-nonlinear - all acoustic properties
%                         'acoustic-linear'   - acoustic properties except
%                                               medium.BonA
%                         'thermal'           - all thermal properties
%
%     'SkullThreshold'
%                   - Threshold parameter (given in Hounsfield units) used
%                     to automatically define a skull mask (default = 250).
%
%     'UseLargestCC'
%                   - Integer controlling how many connected components of
%                     the skull mask is used. This helps to strip out noise
%                     in the CT image, and background structures related to
%                     the CT couch. Can be set to 0 to keep all components
%                     (default = 1).
%                   
% OUTPUTS:
%     medium        - Medium structure used by the k-Wave simulation
%                     functions including the following fields:
%
%                         .density
%                         .sound_speed
%                         .sound_speed_ref
%                         .alpha_coeff
%                         .alpha_power
%                         .BonA
%                         .specific_heat
%                         .thermal_conductivity
%                         .diffusion_coeff_ref
%
%                     If 'ReturnValues' is set to 'acoustic', only the
%                     acoustic properties are return. If set to 'thermal'
%                     only the density and thermal properties are returned.
%                     This allows the medium structure to be passed
%                     directly to the simulation functions.
%
%     skull_mask    - Logical skull mask.
%     head_mask     - Logical head mask.
%     air_mask      - Logical air mask.
%
% ABOUT:
%     author        - Bradley E. Treeby
%     date          - 6th June 2018
%     last update   - 29th May 2020

% =========================================================================
% DEFINE CODE DEFAULTS
% =========================================================================

% define usage defaults
use_hounsfield2density               = false;
plot_conversion                      = false;
return_values                        = 'all';
include_air                          = true;

% define literals used for segmentation
skull_threshold                      = 250;     % [HU]
head_threshold                       = -250;    % [HU]
use_largest_cc                       = 1;
max_hole_size                        = 10;      % [grid points]
min_air_cluster_size                 = 5;       % [grid points]
dilation_size                        = 3;       % [grid points]

% define other literals
scalar_power_law_exponent            = 2;

% =========================================================================
% DEFINE DEFAULT MATERIAL PROPERTY CONSTANTS
% =========================================================================

% define background material properties (water at 30 degrees calculated
% using k-Wave functions)
settings.bg_sound_speed              = 1509;     % [m/s]
settings.bg_density                  = 996;      % [kg/m^3]
settings.bg_alpha_coeff              = 2.17e-3;  % [dB/(MHz^y cm)]
settings.bg_alpha_power              = 2;        % [-]
settings.bg_BonA                     = 5.2;      % [-]
settings.bg_thermal_conductivity     = 0.6;      % [W/(m.K)]
settings.bg_specific_heat            = 4200;     % [J/(kg.K)]

% define soft tissue material properties (used for both skin and brain,
% based on properties of brain from the IT'IS tissue property database
% v4.0) 
settings.tissue_sound_speed          = 1550;     % [m/s]
settings.tissue_density              = 1045;     % [kg/m^3]
settings.tissue_alpha_coeff          = 0.59;     % [dB/(MHz^y cm)]
settings.tissue_alpha_power          = 1.2;      % [-]
settings.tissue_BonA                 = 6.7;      % [-]
settings.tissue_thermal_conductivity = 0.5;      % [W/(m.K)]
settings.tissue_specific_heat        = 3600;     % [J/(kg.K)]

% define air material properties
settings.air_sound_speed             = 343;      % [m/s]
settings.air_density                 = 116;      % [kg/m^3] (inflated by 100)
settings.air_alpha_coeff             = 0;        % [dB/(MHz^y cm)]
settings.air_alpha_power             = 2;        % [-]
settings.air_BonA                    = 0;        % [-]
settings.air_thermal_conductivity    = 0.03;     % [W/(m.K)]
settings.air_specific_heat           = 1000;     % [J/(kg.K)]

% define skull sound speed and density ranges according to Aubry 2003
skull_sound_speed_min                = 1500;     % [m/s]
skull_sound_speed_max                = 3100;     % [m/s]
skull_density_min                    = 1000;     % [kg/m^3]
skull_density_max                    = 2200;     % [kg/m^3]

% define default whole skull attenuation at 1 MHz from Pinton 2012
settings.skull_alpha_coeff           = 13.3;     % [dB/(MHz^y cm)]
settings.skull_alpha_power           = 2;        % [-]

% define skull thermal properties
settings.skull_thermal_conductivity  = 0.32;     % [W/(m.K)]
settings.skull_specific_heat         = 1300;     % [J/(kg.K)] 

% define skull nonlinearity from Renaud 2008
settings.skull_BonA                  = 374;      % [-]

% fit a linear curve between density and sound speed
settings.skull_rho2c_slope           = (skull_sound_speed_max - skull_sound_speed_min) ./ (skull_density_max - skull_density_min);
settings.skull_rho2c_intcp           = skull_sound_speed_min - settings.skull_rho2c_slope * skull_density_min;

% REFERENCES:
%
% J-F. Aubry, M. Tanter, M. Pernot, J. L. Thomas, and M. Fink,
% "Experimental demonstration of noninvasive transskull adaptive
% focusing based on prior computed tomography scans," J. Acoust. Soc.
% Am., 113(1):84-93, 2003.
% 
% G. Pinton, J. F. Aubry, E. Bossy, M. Muller, M. Pernot, and M.
% Tanter, "Attenuation, scattering, and absorption of ultrasound in the
% skull bone," Med. Phys., 39(1), 299-307, 2012.  
%
% Renaud, G., Calle, S., Remenieras, J. P., & Defontaine, M.,
% "Exploration of trabecular bone nonlinear elasticity using time-of-flight
% modulation," IEEE Transactions on Ultrasonics, Ferroelectrics, and
% Frequency Control, 55(7), 1497-1507, 2008.

% =========================================================================
% CHECK INPUTS
% =========================================================================

% check the number of inputs
narginchk(3, inf);

% check input attributes
validateattributes(ct_img, {'numeric'}, {'3d', 'finite'}, 'skull2medium', 'ct_img', 1);
if isempty(ct_calibration)
    use_hounsfield2density = true;
else
    validateattributes(ct_calibration, {'numeric'}, {'2d', 'nrows', 2, 'finite'}, 'skull2medium', 'ct_calibration', 2);
end
validateattributes(f_ref, {'numeric'}, {'scalar', 'finite', 'positive'}, 'skull2medium', 'f_ref', 3);

% check and assign optional inputs
if ~isempty(varargin)
    for input_index = 1:2:length(varargin)
        switch varargin{input_index}               
            case 'ConversionSettings'
                
                % assign inputs
                user_settings = varargin{input_index + 1};
                
                % check defined as a scalar
                if ~isstruct(user_settings)
                    error('Optional input ''Settings'' must be defined as a structure.');
                end
                
                % check and assign values
                settings = assignSettings(settings, user_settings);
                
                % cleanup
                clear user_settings;
                
            case 'HeadThreshold'
                
                % assign input
                head_threshold = varargin{input_index + 1};                
                
                % check properties
                validateattributes(head_threshold, {'numeric'}, {'scalar', 'finite'}, 'skull2medium', '''HeadThreshold''');
                
            case 'IncludeAir'
                
                % assign input
                include_air = varargin{input_index + 1};
                
                % check properties
                validateattributes(include_air, {'logical'}, {'scalar'}, 'skull2medium', '''IncludeAir''');                
                
            case 'Plot'
                
                % assign input
                plot_conversion = varargin{input_index + 1};
                
                % check properties
                validateattributes(plot_conversion, {'logical'}, {'scalar'}, 'skull2medium', '''Plot''');
                
            case 'ReturnValues'
                
                % assign input
                return_values = varargin{input_index + 1};
                
                % check properties
                validateattributes(return_values, {'char'}, {}, 'skull2medium', '''ReturnValues''');
                if ~ismember(return_values, {'all', 'acoustic-linear', 'acoustic-nonlinear', 'thermal'})
                    error('Input ''ReturnValues'' must be set to ''all'', ''acoustic-linear'', ''acoustic-nonlinear'', or ''thermal''.');
                end
                
            case 'SkullThreshold'
                
                % assign input
                skull_threshold = varargin{input_index + 1};                
                
                % check properties
                validateattributes(skull_threshold, {'numeric'}, {'scalar', 'finite'}, 'skull2medium', '''SkullThreshold''');
                
            case 'UseLargestCC'
                
                % assign input
                use_largest_cc = varargin{input_index + 1};
                
                % check properties
                validateattributes(use_largest_cc, {'numeric'}, {'scalar', 'integer', 'nonnegative'}, 'skull2medium', '''UseLargestCC''');
                
            otherwise
                error('Unknown optional input.');
        end
    end
end
       
% =========================================================================
% CREATE MASKS
% =========================================================================

% threshold to find skull and head masks
skull_mask = (ct_img >= skull_threshold);
head_mask  = (ct_img >= head_threshold);

% take only the largest single connected component/s
if use_largest_cc

    % find largest 3D connected component from the skull mask
    skull_cc = bwconncomp(skull_mask);
    numPixels = cellfun(@numel, skull_cc.PixelIdxList);
    [~, idx] = sort(numPixels, 'descend');

    % create new mask with largest components
    if use_largest_cc < length(idx)
        skull_mask = false(size(skull_mask));
        for cc_ind = 1:use_largest_cc
            skull_mask(skull_cc.PixelIdxList{idx(cc_ind)}) = true;
        end
    end
    
    % clean up
    clear skull_cc numPixels idx;
    
    % find largest 3D connected component from the head mask
    head_cc = bwconncomp(head_mask);
    numPixels = cellfun(@numel, head_cc.PixelIdxList);
    [~, idx] = sort(numPixels, 'descend');

    % create new mask with largest components
    if use_largest_cc < length(idx)
        head_mask = false(size(head_mask));
        for cc_ind = 1:use_largest_cc
            head_mask(head_cc.PixelIdxList{idx(cc_ind)}) = true;
        end
    end

    % clean up
    clear head_cc numPixels idx;

end

% fill small holes in the skull mask
skull_mask = fillSmallHoles(skull_mask, max_hole_size);

% fill holes in the head mask
head_mask = fillAllHoles(head_mask, dilation_size);

% find air mask
air_mask = (ct_img < head_threshold);

% fill holes and mask to the air within the head
air_mask = fillAirHoles(air_mask, head_mask, dilation_size, min_air_cluster_size, max_hole_size);

% remove the air from the skull mask
skull_mask = skull_mask & (~air_mask);

% create composite masks (these include the air mask);
ts_mask = head_mask & (~skull_mask);
bg_mask = (~head_mask) & (~skull_mask);

% =========================================================================
% ASSIGN SKULL VALUES
% =========================================================================

if use_hounsfield2density
    
    % use the conversion curve from hounsfield2density, accounting for the
    % scale hounsfield units within this function
     [medium.density, ~] = hounsfield2density(ct_img + 1000);
    
else

    % use the calibration curve between HU and density to calculate density
    % from the CT image, allowing extrapolation outside the parameter range
    medium.density = reshape(...
        interp1(ct_calibration(1, :), ct_calibration(2, :), ct_img(:), 'linear', 'extrap'), ...
        size(ct_img));
    
end

% calculate the sound speed from the density assuming a linear relationship
medium.sound_speed = medium.density .* settings.skull_rho2c_slope + settings.skull_rho2c_intcp;

% define medium property matrics
medium.alpha_coeff          = zeros(size(ct_img));
medium.alpha_power          = zeros(size(ct_img));
medium.thermal_conductivity = zeros(size(ct_img));
medium.specific_heat        = zeros(size(ct_img));
medium.BonA                 = zeros(size(ct_img));

% assign homogeneous values for the absorption and thermal properties
medium.alpha_coeff         (skull_mask) = settings.skull_alpha_coeff;
medium.alpha_power         (skull_mask) = settings.skull_alpha_power;
medium.BonA                (skull_mask) = settings.skull_BonA;
medium.thermal_conductivity(skull_mask) = settings.skull_thermal_conductivity;
medium.specific_heat       (skull_mask) = settings.skull_specific_heat;

% =========================================================================
% ASSIGN SOFT TISSUE AND BACKGROUND VALUES
% =========================================================================

% set background values
medium.sound_speed         (bg_mask) = settings.bg_sound_speed;
medium.density             (bg_mask) = settings.bg_density;
medium.alpha_coeff         (bg_mask) = settings.bg_alpha_coeff;
medium.alpha_power         (bg_mask) = settings.bg_alpha_power;
medium.BonA                (bg_mask) = settings.bg_BonA;
medium.thermal_conductivity(bg_mask) = settings.bg_thermal_conductivity;
medium.specific_heat       (bg_mask) = settings.bg_specific_heat;

% set soft tissue values
medium.sound_speed         (ts_mask) = settings.tissue_sound_speed;
medium.density             (ts_mask) = settings.tissue_density;
medium.alpha_coeff         (ts_mask) = settings.tissue_alpha_coeff;
medium.alpha_power         (ts_mask) = settings.tissue_alpha_power;
medium.BonA                (ts_mask) = settings.tissue_BonA;
medium.thermal_conductivity(ts_mask) = settings.tissue_thermal_conductivity;
medium.specific_heat       (ts_mask) = settings.tissue_specific_heat;

% set air values
if include_air
    medium.sound_speed         (air_mask) = settings.air_sound_speed;
    medium.density             (air_mask) = settings.air_density;
    medium.alpha_coeff         (air_mask) = settings.air_alpha_coeff;
    medium.alpha_power         (air_mask) = settings.air_alpha_power;
    medium.BonA                (air_mask) = settings.air_BonA;
    medium.thermal_conductivity(air_mask) = settings.air_thermal_conductivity;
    medium.specific_heat       (air_mask) = settings.air_specific_heat;
end

% assign reference sound speed and diffusion coefficients to background
% values 
medium.sound_speed_ref     = settings.bg_sound_speed;
medium.diffusion_coeff_ref = settings.bg_thermal_conductivity / (settings.bg_density * settings.bg_specific_heat);

% =========================================================================
% ADJUST ABSORPTION VALUES
% =========================================================================

% account for high-attenuation values and the spatially varying power law
% exponent between skull and background
medium.alpha_coeff = fitPowerLawParamsMulti(medium.alpha_coeff, medium.alpha_power, medium.sound_speed, f_ref, scalar_power_law_exponent);

% assign single power law exponent
medium.alpha_power = scalar_power_law_exponent;

% =========================================================================
% PLOT
% =========================================================================

if plot_conversion
    
    % plot
    figure;

    subplot(3, 3, 1);
    imagesc(ct_img(:, :, (ceil(end/2))));
    colorbar;
    axis image;
    title('CT Image');

    subplot(3, 3, 2);
    imagesc(skull_mask(:, :, (ceil(end/2))));
    colorbar;
    axis image;
    title('Skull Mask');
    
    subplot(3, 3, 3);
    imagesc(head_mask(:, :, (ceil(end/2))));
    colorbar;
    axis image;
    title('Head Mask');
    
    subplot(3, 3, 4);
    imagesc(air_mask(:, :, (ceil(end/2))));
    colorbar;
    axis image;
    title('Air Mask');    
    
    subplot(3, 3, 5);
    imagesc(medium.sound_speed(:, :, (ceil(end/2))));
    colorbar;
    axis image;
    title('Sound Speed');

    subplot(3, 3, 6);
    imagesc(medium.density(:, :, (ceil(end/2))));
    colorbar;
    axis image;
    title('Density');

    subplot(3, 3, 7);
    imagesc(medium.alpha_coeff(:, :, (ceil(end/2))));
    colorbar;
    axis image;
    title('Alpha Coeff');
    
    subplot(3, 3, 8);
    imagesc(medium.thermal_conductivity(:, :, (ceil(end/2))));
    colorbar;
    axis image;
    title('Thermal Conductivity');

    subplot(3, 3, 9);
    imagesc(medium.specific_heat(:, :, (ceil(end/2))));
    colorbar;
    axis image;
    title('Specific Heat');
    
    scaleFig(1.5, 1.5);
    
end

% =========================================================================
% RETURN ONLY SELECTED VALUES
% =========================================================================

switch return_values
    case 'acoustic-nonlinear'
        
        % remove thermal properties
        medium = rmfield(medium, 'thermal_conductivity');
        medium = rmfield(medium, 'specific_heat');
        medium = rmfield(medium, 'diffusion_coeff_ref');
        
    case 'acoustic-linear'
        
        % remove thermal properties
        medium = rmfield(medium, 'thermal_conductivity');
        medium = rmfield(medium, 'specific_heat');
        medium = rmfield(medium, 'diffusion_coeff_ref');
        
        % remove nonlinearity
        medium = rmfield(medium, 'BonA');
        
    case 'thermal'
        
        % remove acoustic properties
        medium = rmfield(medium, 'sound_speed');
        medium = rmfield(medium, 'sound_speed_ref');
        medium = rmfield(medium, 'alpha_coeff');
        medium = rmfield(medium, 'alpha_power');
        medium = rmfield(medium, 'BonA');
        
end

% =========================================================================

function settings = assignSettings(settings, user_settings)
% Nested function to check and assign the user values for the conversion
% settings. Not all values need to be provided (only the values specified
% are used to overwrite the defaults).

% get the field names
field_names = fieldnames(user_settings);
for ind = 1:numel(field_names)
    switch field_names{ind}
        case 'skull_rho2c_slope'

            % assign input
            settings.skull_rho2c_slope = user_settings.skull_rho2c_slope;

            % check input
            validateattributes(settings.skull_rho2c_slope, {'numeric'}, {'scalar', 'finite'}, 'skull2medium', 'settings.skull_rho2c_slope');

        case 'skull_rho2c_intcp'

            % assign input
            settings.skull_rho2c_intcp = user_settings.skull_rho2c_intcp;

            % check input
            validateattributes(settings.skull_rho2c_intcp, {'numeric'}, {'scalar', 'finite'}, 'skull2medium', 'settings.skull_rho2c_intcp');

        case {'bg_sound_speed',          'tissue_sound_speed',          'air_sound_speed', ...
              'bg_density',              'tissue_density',              'air_density', ...
              'bg_thermal_conductivity', 'tissue_thermal_conductivity', 'air_thermal_conductivity', 'skull_thermal_conductivity', ...
              'bg_specific_heat',        'tissue_specific_heat',        'air_specific_heat',        'skull_specific_heat', ...
              'bg_alpha_coeff',          'tissue_alpha_coeff',          'air_alpha_coeff',          'skull_alpha_coeff', ...
              'bg_alpha_power',          'tissue_alpha_power',          'air_alpha_power',          'skull_alpha_power', ...
              'bg_BonA',                 'tissue_BonA',                 'air_BonA',                 'skull_BonA'}
        
            % assign input
            eval(['settings. ' field_names{ind} ' = user_settings.' field_names{ind} ';']);

            % check input
            eval(['validateattributes(settings.' field_names{ind} ', {''numeric''}, {''scalar'', ''nonnegative'', ''finite''}, ''skull2medium'', ''settings.' field_names{ind} ''');']);
            
        otherwise
            error(['The fieldname ' field_names{ind} ' is not a valid input for the optional input ''Settings''.']);
    end
end

% =========================================================================

function bw_img = fillSmallHoles(bw_img, max_hole_size)
% Nested function to fill small holes in a msak based on a threshold
% hole size. This is based directly on the code posted at:
% 
% https://blogs.mathworks.com/steve/2008/08/05/filling-small-holes/

% fill all holes using imfill
filled = imfill(bw_img, 'holes');

% identify the hole pixels using logical operator
holes = filled & ~bw_img;

% use bwareaopen on the holes image to eliminate small holes
bigholes = bwareaopen(holes, max_hole_size^numDim(bw_img));

% use logical operators to identify small holes
smallholes = holes & ~bigholes;

% use a logical operator to fill in the small holes in the original image
bw_img = bw_img | smallholes;

% =========================================================================

function bw_img = fillAllHoles(bw_img, dilat_sz)
% Nested function to fill all holes in a mask.

% dilate image
if numDim(bw_img) == 3
    bw_img = imdilate(bw_img, strel('sphere', dilat_sz));
else
    bw_img = imdilate(bw_img, strel('disk', dilat_sz));
end

% sweeping through 2D planes in each Cartesian direction filling the holes
for layer_ind = 1:size(bw_img, 3)
    bw_img(:, :, layer_ind) = imfill(squeeze(bw_img(:, :, layer_ind)), 'holes');
end

% erode image
if numDim(bw_img) == 3
    bw_img = imerode(bw_img, strel('sphere', dilat_sz));
else
    bw_img = imerode(bw_img, strel('disk', dilat_sz));
end

% =========================================================================

function bw_img = fillAirHoles(bw_img, head_mask, dilation_size, min_air_cluster_size, max_hole_size)
% Nested function to fill holes within the mask.

% find the air internal to the head 
if numDim(head_mask) == 3
    bw_img = bw_img & imerode(head_mask, strel('sphere', dilation_size));
else
    bw_img = bw_img & imerode(head_mask, strel('disk', dilation_size));
end

% remove small pixels, and fill small holes in the air mask
bw_img = bwareaopen(bw_img, min_air_cluster_size^numDim(bw_img));
bw_img = fillSmallHoles(bw_img, max_hole_size^numDim(bw_img));