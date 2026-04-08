import React, { useState, useRef, useEffect } from 'react';
import { Mic, Upload, Sun, Moon } from 'lucide-react';
import { GaussianDFTAudioDetector } from './lib/AudioDetector';
import { GaussianDFTAudioDetectorFast } from './lib/AudioDetectorFast.ts';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface DetectionResults {
  speechPercentage: number;
  duration: number;
  logLikelihood: number[];
  speechFrames: boolean[];
}

function App() {
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'light');

  useEffect(() => {
    localStorage.setItem('theme', theme);
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [regularResults, setRegularResults] = useState<DetectionResults | null>(null);
  const [fastResults, setFastResults] = useState<DetectionResults | null>(null);
  
  const regularDetectorRef = useRef<GaussianDFTAudioDetector | null>(null);
  const fastDetectorRef = useRef<GaussianDFTAudioDetectorFast | null>(null);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;
    
    setFile(selectedFile);
    setIsProcessing(true);
    setError(null);
    setRegularResults(null);
    setFastResults(null);

    try {
      // Initialize detectors if not already created
      if (!regularDetectorRef.current) {
        regularDetectorRef.current = new GaussianDFTAudioDetector();
      }
      
      if (!fastDetectorRef.current) {
        fastDetectorRef.current = new GaussianDFTAudioDetectorFast();
      }

      // Process with regular FFT
      const regularStartTime = performance.now();
      const { speechFrames: regularSpeechFrames, logLikelihood: regularLogLikelihood } = 
        await regularDetectorRef.current.detectSpeech(selectedFile);
      const regularEndTime = performance.now();

      const regularSpeechPercentage = 
        (regularSpeechFrames.filter(Boolean).length / regularSpeechFrames.length) * 100;
      
      setRegularResults({
        speechPercentage: regularSpeechPercentage,
        duration: (regularEndTime - regularStartTime) / 1000,
        logLikelihood: regularLogLikelihood,
        speechFrames: regularSpeechFrames
      });

      // Process with fast FFT
      const fastStartTime = performance.now();
      const { speechFrames: fastSpeechFrames, logLikelihood: fastLogLikelihood } = 
        await fastDetectorRef.current.detectSpeech(selectedFile);
      const fastEndTime = performance.now();

      const fastSpeechPercentage = 
        (fastSpeechFrames.filter(Boolean).length / fastSpeechFrames.length) * 100;
      
      setFastResults({
        speechPercentage: fastSpeechPercentage,
        duration: (fastEndTime - fastStartTime) / 1000,
        logLikelihood: fastLogLikelihood,
        speechFrames: fastSpeechFrames
      });
      
    } catch (error) {
      console.error('Error processing audio:', error);
      setError('Failed to process audio file. Please try again with a different file.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile?.type.startsWith('audio/')) {
      const input = document.getElementById('audio-input') as HTMLInputElement;
      input.files = e.dataTransfer.files;
      handleFileChange({ target: input } as any);
    } else {
      setError('Please drop an audio file.');
    }
  };

  const createChartData = (results: DetectionResults | null) => {
    if (!results) return null;
    
    return {
      labels: Array.from({ length: results.logLikelihood.length }, (_, i) => i),
      datasets: [
        {
          label: 'Log Likelihood Ratio',
          data: results.logLikelihood,
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1
        },
        {
          label: 'Speech Detection',
          data: results.speechFrames.map(v => v ? 1 : 0),
          borderColor: 'rgb(255, 99, 132)',
          tension: 0
        }
      ]
    };
  };

  const regularChartData = createChartData(regularResults);
  const fastChartData = createChartData(fastResults);

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Speech Detection Results'
      }
    },
    scales: {
      y: {
        min: -2,
        max: 2
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:bg-gradient-to-br dark:from-gray-900 dark:to-gray-800 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-gray-900 rounded-xl shadow-xl p-8 max-w-6xl w-full">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center justify-center">
            <Mic className="w-12 h-12 text-indigo-600 dark:text-indigo-400" />
            <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100 ml-3">
              Speech Detector Comparison
            </h1>
          </div>
          <button onClick={toggleTheme} className="focus:outline-none">
            {theme === 'dark' ? (
              <Sun className="w-6 h-6 text-gray-100" />
            ) : (
              <Moon className="w-6 h-6 text-gray-800" />
            )}
          </button>
        </div>

        <div className="space-y-6">
          <div 
            className="relative"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept="audio/*"
              onChange={handleFileChange}
              className="hidden"
              id="audio-input"
              disabled={isProcessing}
            />
            <label
              htmlFor="audio-input"
              className={`
                flex items-center justify-center px-6 py-4 border-2 border-dashed
                rounded-lg cursor-pointer transition-colors dark:bg-gray-800 dark:border-gray-700 dark:text-gray-300
                ${isProcessing
                  ? 'bg-gray-100 border-gray-300 dark:bg-gray-700 dark:border-gray-600'
                  : 'border-indigo-300 hover:border-indigo-400 hover:bg-indigo-50 dark:border-indigo-500 dark:hover:border-indigo-400 dark:hover:bg-indigo-900'}
              `}
            >
              <Upload className="w-6 h-6 text-indigo-500 dark:text-indigo-400 mr-2" />
              <span className="text-gray-600 dark:text-gray-300">
                {file ? file.name : 'Upload or drop audio file'}
              </span>
            </label>
          </div>

          {error && (
            <div className="bg-red-50 dark:bg-red-900 text-red-600 dark:text-red-400 p-4 rounded-lg">
              {error}
            </div>
          )}

          {isProcessing && (
            <div className="text-center py-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mx-auto"></div>
              <p className="text-gray-600 dark:text-gray-300 mt-2">Processing audio with both algorithms...</p>
            </div>
          )}

          {(regularResults || fastResults) && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Regular FFT Results */}
                <div className="bg-indigo-50 dark:bg-indigo-900 rounded-lg p-6 space-y-4">
                  <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-100">Naive DFT (O(n²)) Results</h2>
                  {regularResults && (
                    <div className="space-y-2">
                      <p className="text-gray-600 dark:text-gray-300">
                        Speech detected in{' '}
                        <span className="font-semibold text-indigo-600 dark:text-indigo-400">
                          {regularResults.speechPercentage.toFixed(1)}%
                        </span>
                        of the audio
                      </p>
                      <p className="text-gray-600 dark:text-gray-300">
                        Processing time:{' '}
                        <span className="font-semibold text-indigo-600 dark:text-indigo-400">
                          {regularResults.duration.toFixed(2)}s
                        </span>
                      </p>
                    </div>
                  )}
                </div>

                {/* FFTfast Results */}
                <div className="bg-green-50 dark:bg-green-900 rounded-lg p-6 space-y-4">
                  <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-100">Radix-2 Cooley–Tukey FFT (O(n log(n))) Results</h2>
                  {fastResults && (
                    <div className="space-y-2">
                      <p className="text-gray-600 dark:text-gray-300">
                        Speech detected in{' '}
                        <span className="font-semibold text-green-600 dark:text-green-400">
                          {fastResults.speechPercentage.toFixed(1)}%
                        </span>
                        of the audio
                      </p>
                      <p className="text-gray-600 dark:text-gray-300">
                        Processing time:{' '}
                        <span className="font-semibold text-green-600 dark:text-green-400">
                          {fastResults.duration.toFixed(2)}s
                        </span>
                      </p>
                      {regularResults && (
                        <p className="text-gray-600 dark:text-gray-300">
                          Speed improvement:{' '}
                          <span className="font-semibold text-green-600 dark:text-green-400">
                            {(regularResults.duration / fastResults.duration).toFixed(2)}x
                          </span>
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Regular FFT Chart */}
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm">
                  <h3 className="text-md font-semibold text-gray-800 dark:text-gray-100 mb-2">Naive DFT Visualization</h3>
                  {regularChartData && <Line data={regularChartData} options={{...chartOptions, plugins: {...chartOptions.plugins, title: {...chartOptions.plugins.title, text: 'Regular FFT Results'}}}} />}
                </div>

                {/* FFTfast Chart */}
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm">
                  <h3 className="text-md font-semibold text-gray-800 dark:text-gray-100 mb-2">Radix-2 Cooley–Tukey FFT Visualization</h3>
                  {fastChartData && <Line data={fastChartData} options={{...chartOptions, plugins: {...chartOptions.plugins, title: {...chartOptions.plugins.title, text: 'FFTfast Results'}}}} />}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
