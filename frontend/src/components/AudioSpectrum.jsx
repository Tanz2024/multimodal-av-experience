import { useEffect, useState } from 'react';

export default function AudioSpectrum({ voiceLevel = 0, isActive = false, spectrumData = [] }) {
  const [bars, setBars] = useState(Array(32).fill(0));

  useEffect(() => {
    if (Array.isArray(spectrumData) && spectrumData.length > 0) {
      const nextBars = spectrumData.slice(0, 32).map((v) => Math.max(0, Math.min(100, v * 100)));
      while (nextBars.length < 32) nextBars.push(0);
      setBars(nextBars);
      return;
    }

    if (!isActive) {
      setBars(Array(32).fill(0));
      return;
    }

    const interval = setInterval(() => {
      setBars(prev => prev.map((_, i) => {
        // Simulate spectrum analysis with some randomness
        const baseHeight = voiceLevel * 100;
        const frequency = Math.sin(i * 0.3) * 0.5 + 0.5; // Frequency curve
        const random = Math.random() * 0.3;
        return Math.min(100, baseHeight * frequency + random * 30);
      }));
    }, 50);

    return () => clearInterval(interval);
  }, [isActive, voiceLevel, spectrumData]);

  return (
    <div className="w-full">
      <div className="flex items-end gap-1 h-12">
        {bars.map((height, i) => (
          <div
            key={`bar-${i}`}
            className="w-1.5 rounded-full bg-[#c6a86d] animate-pulse"
            style={{
              height: `${Math.max(4, height)}%`,
              animationDelay: `${i * 0.02}s`
            }}
          />
        ))}
      </div>
      <div className="mt-2 text-[11px] uppercase tracking-[0.18em] text-[#7b8088]">
        {isActive ? 'Audio Activity' : 'Idle'}
      </div>
    </div>
  );
}
