load("datasets/testset.mat");
speeds_of_sound = double(squeeze(speeds_of_sound(:,1,:,:)));

% Adding parameters
% TODO: this should be dynamic from the model
omega = 1.;
source_location = [82, 48];
pml_size = 8;
sigma_star = 2;

% Making source maps
sos_map = squeeze(speeds_of_sound(1,:,:));

P = zeros(size(speeds_of_sound,1), ...
    size(speeds_of_sound,2),size(speeds_of_sound,3));


parpool(8)
parfor samplenum = 1:64 %size(speeds_of_sound,1)
    sos_map = squeeze(speeds_of_sound(samplenum,:,:));
    p = kwave_solver(sos_map,source_location,omega,1.,samplenum);
    P(samplenum,:,:) = reshape(p, 96,96);
    pause(1)
end
delete(gcp)


save("kwave_results.mat","P")