#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>

#define FRAME_SIZE 1024
#define HOP_SIZE 512

void spectral_subtraction(double *audio_data, int num_frames, double *cleaned_audio) {
    fftw_complex *stft_in, *stft_out;
    fftw_plan plan_forward, plan_backward;
    double *magnitude = (double *)malloc(FRAME_SIZE * sizeof(double));
    double *noise_estimation = (double *)calloc(FRAME_SIZE, sizeof(double));

    stft_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * FRAME_SIZE);
    stft_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * FRAME_SIZE);
    plan_forward = fftw_plan_dft_r2c_1d(FRAME_SIZE, audio_data, stft_in, FFTW_ESTIMATE);
    plan_backward = fftw_plan_dft_c2r_1d(FRAME_SIZE, stft_out, cleaned_audio, FFTW_ESTIMATE);

    for (int i = 0; i < num_frames; i += HOP_SIZE) {
        fftw_execute(plan_forward);

        for (int j = 0; j < FRAME_SIZE; ++j) {
            double real = stft_in[j][0];
            double imag = stft_in[j][1];
            magnitude[j] = sqrt(real * real + imag * imag);
        }

        for (int j = 0; j < FRAME_SIZE; ++j) {
            noise_estimation[j] += magnitude[j];
        }
    }

    for (int j = 0; j < FRAME_SIZE; ++j) {
        noise_estimation[j] /= (num_frames / HOP_SIZE);
    }

    for (int i = 0; i < num_frames; i += HOP_SIZE) {
        fftw_execute(plan_forward);

        for (int j = 0; j < FRAME_SIZE; ++j) {
            double real = stft_in[j][0];
            double imag = stft_in[j][1];
            magnitude[j] = fmax(0.0, sqrt(real * real + imag * imag) - noise_estimation[j]);

            stft_out[j][0] = magnitude[j] * cos(atan2(imag, real));
            stft_out[j][1] = magnitude[j] * sin(atan2(imag, real));
        }

        fftw_execute(plan_backward);
    }

    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
    fftw_free(stft_in);
    fftw_free(stft_out);
    free(magnitude);
    free(noise_estimation);
}

double* read_wav_file(const char *filename, int *num_samples) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file %s\n", filename);
        return NULL;
    }

    fseek(file, 44, SEEK_SET);

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 44, SEEK_SET);

    *num_samples = (file_size - 44) / sizeof(short);
    short *buffer = (short *)malloc(*num_samples * sizeof(short));

    fread(buffer, sizeof(short), *num_samples, file);
    fclose(file);

    double *audio_data = (double *)malloc(*num_samples * sizeof(double));
    for (int i = 0; i < *num_samples; i++) {
        audio_data[i] = (double)buffer[i] / 32768.0; // normalize to [-1, 1]
    }

    free(buffer);
    return audio_data;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input_wav_file>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    int num_samples;
    double *audio_data = read_wav_file(input_file, &num_samples);

    if (audio_data == NULL) {
        printf("Error reading WAV file.\n");
        return 1;
    }

    double *cleaned_audio = (double *)malloc(num_samples * sizeof(double));

    spectral_subtraction(audio_data, num_samples, cleaned_audio);

    printf("Cleaned audio signal:\n");
    for (int i = 0; i < num_samples; i++) {
        printf("%f ", cleaned_audio[i]);
    }

    free(audio_data);
    free(cleaned_audio);
    return 0;
}
