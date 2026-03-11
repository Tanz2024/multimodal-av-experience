import { useEffect, useRef } from 'react';

export default function VoiceWave({ isActive, amplitude = 0.5, spectrumData = [] }) {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const phaseRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    };
    
    resize();
    window.addEventListener('resize', resize);

    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      const centerY = height / 2;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!isActive) {
        // Draw flat line when not active
        ctx.strokeStyle = 'rgba(198, 168, 109, 0.3)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        ctx.lineTo(width, centerY);
        ctx.stroke();
        return;
      }

      // Draw animated wave
      const bins = Array.isArray(spectrumData) ? spectrumData : [];
      const quarter = Math.max(1, Math.floor(bins.length / 4));
      const low = bins.slice(0, quarter);
      const high = bins.slice(Math.max(0, bins.length - quarter));
      const avg = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
      const bassEnergy = Math.max(0, Math.min(1, avg(low)));
      const trebleEnergy = Math.max(0, Math.min(1, avg(high)));
      const dynamicAmp = Math.max(0.06, Math.min(1.0, (amplitude * 0.65) + (bassEnergy * 0.9)));
      const dynamicFreq = 3.5 + (trebleEnergy * 3.0);
      phaseRef.current += 0.04 + (trebleEnergy * 0.08);
      
      const gradient = ctx.createLinearGradient(0, 0, width, 0);
      gradient.addColorStop(0, 'rgba(198, 168, 109, 0.3)');
      gradient.addColorStop(0.5, 'rgba(198, 168, 109, 1)');
      gradient.addColorStop(1, 'rgba(198, 168, 109, 0.3)');
      
      ctx.strokeStyle = gradient;
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      // Draw main wave
      ctx.beginPath();
      for (let x = 0; x < width; x += 2) {
        const normalizedX = x / width;
        const waveAmplitude = dynamicAmp * 30;
        const y = centerY + Math.sin(normalizedX * Math.PI * dynamicFreq + phaseRef.current) * waveAmplitude;
        
        if (x === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Draw secondary wave (offset)
      ctx.strokeStyle = 'rgba(216, 187, 130, 0.4)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let x = 0; x < width; x += 2) {
        const normalizedX = x / width;
        const waveAmplitude = dynamicAmp * 20;
        const y = centerY + Math.sin(normalizedX * Math.PI * dynamicFreq + phaseRef.current + Math.PI / 2) * waveAmplitude;
        
        if (x === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      animationRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      window.removeEventListener('resize', resize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive, amplitude, spectrumData]);

  return (
    <div className="w-full h-12 rounded-lg border border-[#e3ddd2] bg-white/70 px-2 py-1">
      <canvas ref={canvasRef} className="w-full h-full" />
    </div>
  );
}
