function allTFeatures = tamura3sigs_no_stats(Im)
% MATLAB-toolbox-independent equivalent of reference/Tamura3Sigs.m.
% It preserves its three formulas; only Statistics Toolbox kurtosis is
% replaced by its fourth-central-moment definition.
Im = rgb2gray(Im);
if ~isa(Im,'double'), Im = im2double(Im); end
allTFeatures = [tamura_coarseness(Im), tamura_contrast(Im), tamura_directionality(Im)];
end

function Fdir = tamura_directionality(Im)
[gx,gy] = gradient(Im); [t,r] = cart2pol(gx,gy);
good = find(r > .15 .* max(r(:)));
if isempty(good), Fdir = 0; return; end
t = t(good);
Fdir = 1/(entropy(t)+1);
end

function Fc = tamura_contrast(Im)
x = Im(:); s = std(x);
if abs(s) < 1e-10, Fc = 0; return; end
mu = mean(x);
pearson_kurtosis = mean((x-mu).^4) / mean((x-mu).^2)^2;
Fc = s / ((pearson_kurtosis / s^4)^(.25));
end

function Fc = tamura_coarseness(Im)
kk = 0:6;
Hdelta = zeros(size(Im,1), size(Im,2), numel(kk));
Vdelta = Hdelta;
for ii = 1:kk(end)
    A = conv2(Im, ones(2.^kk(ii))./(2.^kk(ii)).^2, 'same');
    shift = 2.^kk(ii);
    plus = zeros(size(A)); minus = zeros(size(A));
    plus(:,1:end-shift+1) = A(:,shift:end); minus(:,shift:end) = A(:,1:end-shift+1);
    Hdelta(:,:,ii) = abs(plus-minus);
    plus = zeros(size(A)); minus = zeros(size(A));
    plus(1:end-shift+1,:) = A(shift:end,:); minus(shift:end,:) = A(1:end-shift+1,:);
    Vdelta(:,:,ii) = abs(plus-minus);
end
hij = reshape(Hdelta, [], size(Hdelta,3));
vij = reshape(Vdelta, [], size(Vdelta,3));
best = zeros(size(hij,1),1);
for ii = 1:size(hij,1)
    [hm, hi] = max(hij(ii,:)); [vm, vi] = max(vij(ii,:));
    if hm >= vm, best(ii) = kk(hi); else, best(ii) = kk(vi); end
end
Fc = mean(best);
end
