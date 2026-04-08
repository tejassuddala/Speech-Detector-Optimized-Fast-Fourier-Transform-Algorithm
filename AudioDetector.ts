import { FFT } from './FFT';

export class GaussianDFTAudioDetector {
  private frameLength: number;
  private hopLength: number;
  private sampleRate: number;
  private fft: FFT;

  constructor(frameLength = 1024, hopLength = 512, sampleRate = 16000) {
    this.frameLength = frameLength;
    this.hopLength = hopLength;
    this.sampleRate = sampleRate;
    this.fft = new FFT(frameLength);
  }

  private async getAudioData(file: File): Promise<Float32Array> {
    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioContext = new AudioContext({ sampleRate: this.sampleRate });
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      return audioBuffer.getChannelData(0);
    } catch (error) {
      throw new Error('Failed to decode audio file. Please ensure it\'s a valid audio format.');
    }
  }

  private frameSignal(signal: Float32Array): Float32Array[] {
    if (signal.length < this.frameLength) {
      throw new Error('Audio file is too short for analysis.');
    }

    const frames: Float32Array[] = [];
    for (let i = 0; i < signal.length - this.frameLength; i += this.hopLength) {
      const frame = signal.slice(i, i + this.frameLength);
      frames.push(frame);
    }
    return frames;
  }

  private applyWindow(frame: Float32Array): Float32Array {
    // Hanning window
    const window = new Float32Array(frame.length);
    for (let i = 0; i < frame.length; i++) {
      window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (frame.length - 1)));
    }
    return frame.map((x, i) => x * window[i]);
  }

  private extractDFTFeatures(frames: Float32Array[]): Float32Array[] {
    return frames.map(frame => {
      const windowedFrame = this.applyWindow(frame);
      const dft = this.fft.forward(windowedFrame);
      return new Float32Array(dft.slice(0, this.frameLength / 2 + 1));
    });
  }

  private estimateNoiseVariance(features: Float32Array[], nNoiseFrames = 10): Float32Array {
    const numFramesToUse = Math.min(features.length, nNoiseFrames);
    if (numFramesToUse === 0) {
      throw new Error('Not enough frames for noise estimation.');
    }

    const variance = new Float32Array(features[0].length);
    
    for (let i = 0; i < numFramesToUse; i++) {
      for (let j = 0; j < variance.length; j++) {
        variance[j] += Math.pow(features[i][j], 2);
      }
    }

    return variance.map(v => v / numFramesToUse);
  }

  private estimatePrioriSNR(features: Float32Array[], sigmaN: Float32Array, alpha = 0.98): Float32Array[] {
    const xi: Float32Array[] = [];
    
    // Initial estimate
    const firstFrame = features[0];
    const gamma = firstFrame.map((y, j) => Math.pow(y, 2) / (sigmaN[j] || Number.EPSILON));
    xi.push(new Float32Array(gamma.map(g => Math.max(g - 1, 0))));

    // Estimate for remaining frames
    for (let m = 1; m < features.length; m++) {
      const frame = features[m];
      const prevXi = xi[m - 1];
      const newXi = new Float32Array(frame.length);

      for (let j = 0; j < frame.length; j++) {
        const sigmaNJ = sigmaN[j] || Number.EPSILON;
        const ampPrev = Math.sqrt(prevXi[j] * sigmaNJ);
        const gammaCurrent = Math.pow(frame[j], 2) / sigmaNJ;
        newXi[j] = alpha * (Math.pow(ampPrev, 2) / sigmaNJ) + 
                   (1 - alpha) * Math.max(gammaCurrent - 1, 0);
      }

      xi.push(newXi);
    }

    return xi;
  }

  public async detectSpeech(file: File, threshold = 0.5): Promise<{
    speechFrames: boolean[];
    logLikelihood: number[];
  }> {
    if (!file.type.startsWith('audio/')) {
      throw new Error('Please provide a valid audio file.');
    }

    const audio = await this.getAudioData(file);
    const frames = this.frameSignal(audio);
    const features = this.extractDFTFeatures(frames);
    
    const sigmaN = this.estimateNoiseVariance(features);
    const xi = this.estimatePrioriSNR(features, sigmaN);
    
    const logLikelihood: number[] = [];
    const speechFrames: boolean[] = [];

    for (let m = 0; m < features.length; m++) {
      const frame = features[m];
      let sumLogL = 0;

      for (let j = 0; j < frame.length; j++) {
        const sigmaS = xi[m][j] * sigmaN[j];
        const y2 = Math.pow(frame[j], 2);
        const gamma = y2 / (sigmaN[j] || Number.EPSILON);
        const xiJ = sigmaS / (sigmaN[j] || Number.EPSILON);
        
        const L = (1 / (1 + xiJ)) * Math.exp((gamma * xiJ) / (1 + xiJ));
        sumLogL += Math.log(Math.max(L, Number.EPSILON));
      }

      const avgLogL = sumLogL / frame.length;
      logLikelihood.push(avgLogL);
      speechFrames.push(avgLogL > Math.log(threshold));
    }

    return { speechFrames, logLikelihood };
  }
}