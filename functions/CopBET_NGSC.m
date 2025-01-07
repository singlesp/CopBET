function out = CopBET_NGSC(in, atlas_file, num_workers, varargin)
% Copenhagen Brain Entropy Toolbox: NGSC (Normalized Geodesic Spectral Clustering)
% Calculates NGSC wholebrain and per parcel as defined by Siegel et al., 2024.
%
% Input:
%   in: table where the first column contains paths to .nii files
%   atlas_file: path to the atlas .nii file
%   num_workers: Whether to run calculations in parallel, and if so how many
%
% Output:
%   out: table containing NGSC values (whole brain and per ROI)

% Set default value for num_workers if not provided
if nargin < 3
    num_workers = 1;
end

% Setup parallel pool if needed
if num_workers > 1
    pool = gcp('nocreate');
    if isempty(pool)
        parpool(num_workers);
    elseif pool.NumWorkers ~= num_workers
        delete(pool);
        parpool(num_workers);
    end
end

% Initialize output table
out = in;

[out, ~, in, ~] = CopBET_function_init(in, varargin);

% Read atlas
atlas = niftiread(atlas_file);
atlas = reshape(atlas, [], 1);
Vparc(:, 1) = atlas;
roi = unique(Vparc);
roi = roi(roi ~= 0);
num_rois = length(roi);

% Initialize output arrays
ngsc_entropy_global = zeros(height(in), 1);
ngsc_entropy_regional = cell(height(in), 1);

% Main processing loop
parfor (ses = 1:height(in), num_workers)
    disp(['Working on NGSC calculations for session: ', num2str(ses)])
    
    % Load 4D file
    path = in{ses,1}{1};
    V = double(niftiread(path));
    V1 = reshape(V, [], size(V, 4));
    
    % Identify and remove unwanted voxels
    zero_atlas = find(Vparc == 0);
    zero_ts = find(all(V1 == 0, 2));
    voxels_to_remove = unique([zero_atlas; zero_ts]);
    
    V_nonzero = V1;
    V_nonzero(voxels_to_remove, :) = [];
    Vparc_ses = Vparc;
    Vparc_ses(voxels_to_remove, :) = [];
    
    % Standardize data if necessary
    mean_threshold = 1e-6;
    std_threshold = 0.01;
    means = mean(V_nonzero, 2);
    stds = std(V_nonzero, 0, 2);
    
    if all(abs(means) < mean_threshold) && all(abs(stds - 1) < std_threshold)
        disp('Data appears to be already standardized.');
    else
        disp('Standardizing data...');
        V_nonzero = (V_nonzero - means) ./ stds;
    end
    
    % Whole brain NGSC
    [~, ~, eigenvalues] = pca(V_nonzero');
    m = length(eigenvalues);
    normalizedEigenvalues = eigenvalues / sum(eigenvalues);
    normalizedEigenvalues(normalizedEigenvalues == 0) = eps;
    ngsc_entropy_global(ses) = -sum(normalizedEigenvalues .* log(normalizedEigenvalues)) / log(m);
    
    % Regional NGSC
    ngsc_roi = zeros(1, num_rois);
    for roi_idx = 1:num_rois
        reg = roi(roi_idx);
        Vparc_roi = find(Vparc_ses == reg);
        regionData = V_nonzero(Vparc_roi, :);
        
        [~, ~, eigenvalues] = pca(regionData');
        m = length(eigenvalues);
        normalizedEigenvalues = eigenvalues / sum(eigenvalues);
        normalizedEigenvalues(normalizedEigenvalues == 0) = eps;
        ngsc_roi(roi_idx) = -sum(normalizedEigenvalues .* log(normalizedEigenvalues)) / log(m);
    end
    
    ngsc_entropy_regional{ses} = ngsc_roi;
end

% Add results to output table
out.ngsc_entropy_global = ngsc_entropy_global;
out.ngsc_entropy_regional = ngsc_entropy_regional;

end