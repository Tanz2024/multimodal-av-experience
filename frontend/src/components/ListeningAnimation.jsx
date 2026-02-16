import { useEffect, useState } from 'react';

export default function ListeningAnimation({ isListening, voiceLevel = 0 }) {
  const [pulseScale, setPulseScale] = useState(1);

  useEffect(() => {
    if (!isListening) {
      setPulseScale(1);
      return;
    }

    const interval = setInterval(() => {
      setPulseScale(1 + (Math.random() * 0.3 + voiceLevel * 0.5));
    }, 150);

    return () => clearInterval(interval);
  }, [isListening, voiceLevel]);

  if (!isListening) return null;

  return (
    <div className="fixed inset-0 z-40 pointer-events-none flex items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <div className="relative h-36 w-36">
          <div
            className="absolute inset-0 rounded-full bg-[#c6a86d]/10"
            style={{ transform: `scale(${pulseScale})` }}
          />
          <div
            className="absolute inset-3 rounded-full bg-[#c6a86d]/20 animate-ping"
            style={{ animationDuration: '1.4s' }}
          />
          <div
            className="absolute inset-6 rounded-full bg-[#c6a86d]/30"
            style={{ transform: `scale(${pulseScale * 0.7})` }}
          />
          <div className="absolute inset-9 rounded-full bg-[#c6a86d]" />
        </div>
        <div className="text-xs uppercase tracking-[0.32em] text-[#7b8088]">
          Listening...
        </div>
      </div>
    </div>
  );
}
