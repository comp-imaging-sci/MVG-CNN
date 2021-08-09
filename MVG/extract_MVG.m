% MATLAB Scripts for computing multiplex visibility graph on 10-s GCAMP data 
clc;
clear;
% configure fast_NVG
addpath fast_NVG/;
% Load Paxinos atlas provided
load atlas.mat;
%%
% define the root data directory where folders of each mouse is located
root_fid = './path/to/main/data/directory';
% name of folders containing recordings for individual mouse
mouse_list = ["191114", "191115", "191204", "191211","191218","200127", "200128", "200204", "200313","200402", "200813", "200814", "200910", "201002"];
% define which parcels to use, here used common shared parcels for all subjects
% self-defined is allow in the function "define-rois.m"
roi_names = define_rois("wholebrain");
num_rois = numel(roi_names);
num_mouse = numel(mouse_list);
% epoch duration in unit of second
time = 10;
% derive number of frames for such epoch duration
num_frames = floor(time*168/10);

%% compute MVG
for mouse_idx = 1:num_mouse
    fid = fullfile(root_fid, mouse_list(mouse_idx));
    fnames = dir(fid);
for file_idx=3:numel(fnames)
    [~,prefix,~] = fileparts(fnames(file_idx).name);
    fname=fullfile(fnames(file_idx).folder, fnames(file_idx).name);
    disp(fname);  load(fname);
    gcamp6 = data(:,:,1:num_frames,:);
    num_epochs = size(gcamp6,4);
    % define the overlapped region between the atlas and the brain mask
    mask_atlas = AtlasSeeds.*xform_mask;
    
    for epoch=1:num_epochs
        data = gcamp6(:,:,:,epoch);
        label = scoringindex_file(epoch);
        [toi] = parcel2trace(data, mask_atlas, roi_names);
        am = zeros(num_rois, num_frames, num_frames);
        
        for roi_idx = 1:size(toi,1)
            [VG] = fast_NVG(toi(roi_idx, :), 1:num_frames);
            am(roi_idx,:,:) = VG;      
        end          
        
        % raw label: 0=wake, 1=NREM, 2=artifacts, 3=REM
        if label ~= 2 % artifacts are not used in this study
            if label == 3 
                % redefine the label as: 0=wake, 1=NREM and 2=REM
                label = 2;
            end
          
            % save MVG for each data epoch in mat. file
            save(sprintf('./path/to/main/data/directory/%s-MVG-%ds/%s_epoch%d.mat', mouse_list(mouse_idx), time, prefix, epoch), 'am', 'label');
        end
    end
end
end
