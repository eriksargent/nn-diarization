scale = 10;

files = dir('Sound Files/*.wav');
for file = files'
    disp(file);
    inname = sprintf('%s%s', 'Sound Files/', file.name);
    outname = sprintf('%s%s', 'Downsampled/', file.name);
    [y, Fs] = audioread(inname);
    T = 1 / Fs;
    cutoff = 1000;
    t = 1 / (1000 * 2 * pi);
    a = T/t;
    filtered = filter(a, [1 a-1], y);
    downsampled = downsample(filtered, scale);
    audiowrite(outname, downsampled, Fs / scale)
end