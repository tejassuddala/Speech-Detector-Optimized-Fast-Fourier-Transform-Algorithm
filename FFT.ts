export class FFT {
  private size: number;
  private cosTable: Float32Array;
  private sinTable: Float32Array;

  constructor(size: number) {
    if (size <= 0 || (size & (size - 1)) !== 0) {
      throw new Error('FFT size must be a positive power of 2');
    }

    this.size = size;
    this.cosTable = new Float32Array(size);
    this.sinTable = new Float32Array(size);

    for (let i = 0; i < size; i++) {
      const angle = (2 * Math.PI * i) / size;
      this.cosTable[i] = Math.cos(angle);
      this.sinTable[i] = Math.sin(angle);
    }
  }

  public forward(input: Float32Array): Float32Array {
    if (input.length !== this.size) {
      throw new Error(`Input size must match FFT size (${this.size})`);
    }

    const real = new Float32Array(this.size);
    const imag = new Float32Array(this.size);
    const magnitude = new Float32Array(this.size);

    // Copy input to real array
    input.forEach((val, i) => real[i] = val);

    // FFT
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        const angle = (2 * Math.PI * i * j) / this.size;
        real[i] += input[j] * Math.cos(angle);
        imag[i] -= input[j] * Math.sin(angle);
      }
    }

    // Calculate magnitude
    for (let i = 0; i < this.size; i++) {
      magnitude[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
    }

    return magnitude;
  }
}