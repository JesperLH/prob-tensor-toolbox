function [sse,varexpl] = calcFit(data,model)

    sse = data-nmodel(model);
    sse = nansum(sse(:).^2);
    sst = nansum(data(:).^2);

    varexpl = 1-sse/sst;
end