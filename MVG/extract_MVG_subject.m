% MVG pre-processing classwise for testing
clc;
clear;
addpath fast_NVG/;

load paxinos_atlas.mat;
root_fid = '/shared/anastasio-s1/MRI/xiaohui/mouse_optical/sleep-stage/2020-G5/';

% files and paths
mouse_list = ["191114", "191115", "191204", "191211","191218","200127", "200128", "200204", "200313","200402", "200813", "200814", "200910", "201002"];
roi_names = define_rois("wholebrain");
num_rois = numel(roi_names);
num_mouse = numel(mouse_list);
time = 10;
num_frames = floor(time*168/10);

%roi_names = [25, 29, 37];
%roi_names = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19];
% roi_names = [22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39];
% roi_names = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,...
%   24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39];

%% compute TOIs
for mouse_idx = 1:num_mouse
    fid = fullfile(root_fid, mouse_list(mouse_idx));
    fnames = dir(fid);
for file_idx=3:numel(fnames)
    [~,prefix,~] = fileparts(fnames(file_idx).name);
    fname=fullfile(fnames(file_idx).folder, fnames(file_idx).name);
    disp(fname);  load(fname);
    %save(sprintf('/shared/anastasio-s1/MRI/xiaohui/mouse_optical/sleep-stage/2020-G5/%s-MVG/%s_true_scoring.mat', mouse_list(mouse_idx), prefix), 'scoringindex_file');
    gcamp6 = data(:,:,1:num_frames,:);
    num_epochs = size(gcamp6,4);
    mask_atlas = AtlasSeeds.*xform_mask;
    
    for epoch=1:num_epochs
        data = gcamp6(:,:,:,epoch);
        label = scoringindex_file(epoch);
        [toi] = parcel2trace_MVG(data, mask_atlas, roi_names);
        am = zeros(num_rois, num_frames, num_frames);
        
        for roi_idx = 1:size(toi,1)
            [VG] = fast_NVG(toi(roi_idx, :), 1:num_frames);
            am(roi_idx,:,:) = VG;
            %[HVG] = fast_HVG(toi(roi_idx, :), 1:168);
            %am(roi_idx,:,:) = HVG;           
        end          
        
        if label ~= 2
            if label == 3
                label = 2;
            end
            %disp(label);
            save(sprintf('/shared/anastasio-s1/MRI/xiaohui/mouse_optical/sleep-stage/2020-G5/%s-MVG-%ds/%s_epoch%d.mat', mouse_list(mouse_idx), time, prefix, epoch), 'am', 'label');
        end
    end
end
end