function [roi_names] = define_rois(area_name)
% function to define ROIs used in computing MVG
if area_name == "wholebrain"
    roi_names = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39];
elseif area_name == "left_hemis"
    roi_names = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19];
elseif area_name == "right_hemis"
    roi_names = [22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39];
else
   roi_names = str2double(area_name);
end