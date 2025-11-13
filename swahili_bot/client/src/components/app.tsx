import React, { useState } from 'react';
import './app.css';

type ConnectionState = 'idle' | 'connecting' | 'connected' | 'error';

export default function App() {
  const [connectionState, setConnectionState] = useState<ConnectionState>('idle');
  const [speakerId, setSpeakerId] = useState<number>(22);
  const [message, setMessage] = useState<string>('');

  const connect = async () => {
    setConnectionState('connecting');
    setMessage('Inaunganisha kwa huduma...');

    try {
      // Test connection to backend
      const response = await fetch('/offer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sdp: 'test',  // Placeholder
          speaker_id: speakerId,
        }),
      });

      if (response.ok) {
        setConnectionState('connected');
        setMessage(`Imeunganishwa! Speaker ID: ${speakerId}`);
      } else {
        throw new Error('Connection failed');
      }
    } catch (error) {
      setConnectionState('error');
      setMessage('Hitilafu wakati wa kuunganisha. Tafadhali jaribu tena.');
      console.error('Connection error:', error);
    }
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1 className="title">🎤 Rafiki</h1>
          <p className="subtitle">Msaidizi Wako wa Kiswahili</p>
        </header>

        <div className="controls">
          <div className="speaker-control">
            <label htmlFor="speaker-id">
              Sauti (Speaker ID):
              <input
                id="speaker-id"
                type="number"
                min="0"
                max="100"
                value={speakerId}
                onChange={(e) => setSpeakerId(parseInt(e.target.value) || 22)}
                disabled={connectionState === 'connected'}
                className="speaker-input"
              />
            </label>
            <span className="speaker-hint">
              Badilisha nambari ili kubadilisha sauti (kwa mfano: 22, 25, 30)
            </span>
          </div>

          <button
            onClick={connect}
            disabled={connectionState === 'connecting'}
            className={`connect-button ${connectionState === 'connected' ? 'connected' : ''}`}
          >
            {connectionState === 'idle' && '🎙️ Angalia Muunganisho'}
            {connectionState === 'connecting' && '⏳ Inaunganisha...'}
            {connectionState === 'connected' && '✅ Imeunganishwa'}
            {connectionState === 'error' && '🔄 Jaribu Tena'}
          </button>
        </div>

        <div className="transcript-container">
          <h2 className="transcript-title">Hali</h2>
          <div className="transcript">
            {message ? (
              <p className="transcript-line">{message}</p>
            ) : (
              <p className="transcript-empty">
                Bonyeza "Angalia Muunganisho" ili kuangalia kama huduma inaendesha.
              </p>
            )}
          </div>
        </div>

        <div className="info">
          <h3>📚 Maelekezo</h3>
          <ul>
            <li>Huduma hii inatumia teknolojia ya Pipecat na Modal</li>
            <li>Huduma za AI: Omnilingual ASR, Aya-101, na Swahili CSM TTS</li>
            <li>Speaker ID inabadilisha sauti ya TTS</li>
            <li>Ili kutumia mazungumzo ya sauti, unahitaji kuwa na WebRTC support</li>
          </ul>

          <div className="service-status">
            <h4>Huduma Zilizosawazishwa:</h4>
            <ul>
              <li>✅ STT: Omnilingual ASR (swh_Latn)</li>
              <li>✅ LLM: Aya-101 (13B parameters)</li>
              <li>✅ TTS: Swahili CSM-1B</li>
            </ul>
          </div>
        </div>

        <footer className="footer">
          <p>
            Powered by{' '}
            <a href="https://modal.com" target="_blank" rel="noopener noreferrer">
              Modal
            </a>
            {' '}&amp;{' '}
            <a href="https://pipecat.ai" target="_blank" rel="noopener noreferrer">
              Pipecat
            </a>
          </p>
          <p className="note">
            <strong>Kumbuka:</strong> Hii ni toleo la kuonyesha huduma. Kwa mazungumzo ya sauti kamili,
            fungua app kutoka kwa Modal URL yako.
          </p>
        </footer>
      </div>
    </div>
  );
}
