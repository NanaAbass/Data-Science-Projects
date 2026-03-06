import { useState, useRef, useEffect } from "react";

const SYSTEM_PROMPT = `You are a medical test recommendation assistant integrated into a portfolio project that demonstrates NLP-based recommender systems.

Given a user's symptom description, respond with a JSON object (no markdown, no explanation, just raw JSON) in this exact format:
{
  "disclaimer": "One sentence educational disclaimer",
  "recommendations": [
    {
      "rank": 1,
      "condition": "Possible condition name",
      "similarity_score": 0.87,
      "model_used": "Ensemble (TF-IDF + Word2Vec)",
      "tests": ["Test 1", "Test 2", "Test 3"],
      "urgency": "routine|soon|urgent",
      "reasoning": "Brief 1-sentence explanation of why these tests are recommended"
    },
    {
      "rank": 2,
      "condition": "Another possible condition",
      "similarity_score": 0.73,
      "model_used": "TF-IDF",
      "tests": ["Test A", "Test B"],
      "urgency": "routine|soon|urgent",
      "reasoning": "Brief explanation"
    },
    {
      "rank": 3,
      "condition": "Third possible condition",
      "similarity_score": 0.61,
      "model_used": "Word2Vec",
      "tests": ["Test X"],
      "urgency": "routine|soon|urgent",
      "reasoning": "Brief explanation"
    }
  ],
  "top_tests": ["Most common test across results", "Second test", "Third test"],
  "symptom_keywords": ["keyword1", "keyword2", "keyword3", "keyword4"]
}

Use realistic medical test names (Full Blood Count, HbA1c, Liver Function Test, etc.). Make urgency realistic based on symptoms. Always include exactly 3 recommendations.`;

const URGENCY_CONFIG = {
  urgent:  { color: "#ff4757", bg: "rgba(255,71,87,0.12)",  label: "URGENT",  icon: "🚨" },
  soon:    { color: "#ffa502", bg: "rgba(255,165,2,0.12)",   label: "SEE DOCTOR SOON", icon: "⚠️" },
  routine: { color: "#2ed573", bg: "rgba(46,213,115,0.12)",  label: "ROUTINE", icon: "✅" },
};

const SAMPLE_SYMPTOMS = [
  "I feel very tired all the time, my skin looks pale and I get breathless easily",
  "Excessive thirst, frequent urination, blurry vision and fatigue",
  "Severe chest pain radiating to my left arm with sweating and nausea",
  "My joints are stiff and swollen in the morning, especially in my hands",
  "Persistent cough with blood, night sweats and unexplained weight loss",
  "Burning sensation when urinating and I need to go very frequently",
];

function TypewriterText({ text }) {
  const [displayed, setDisplayed] = useState("");
  useEffect(() => {
    setDisplayed("");
    let i = 0;
    const timer = setInterval(() => {
      if (i < text.length) {
        setDisplayed(text.slice(0, i + 1));
        i++;
      } else {
        clearInterval(timer);
      }
    }, 18);
    return () => clearInterval(timer);
  }, [text]);
  return <span>{displayed}</span>;
}

function ScoreBar({ score }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{
        flex: 1, height: 6, background: "rgba(255,255,255,0.08)",
        borderRadius: 3, overflow: "hidden"
      }}>
        <div style={{
          height: "100%", width: `${score * 100}%`,
          background: `linear-gradient(90deg, #6c63ff, #a78bfa)`,
          borderRadius: 3, transition: "width 0.8s ease"
        }} />
      </div>
      <span style={{ fontSize: 12, color: "#a78bfa", fontFamily: "monospace", minWidth: 36 }}>
        {(score * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function KeywordChip({ word }) {
  return (
    <span style={{
      padding: "3px 10px", borderRadius: 20,
      background: "rgba(108,99,255,0.15)",
      border: "1px solid rgba(108,99,255,0.3)",
      color: "#a78bfa", fontSize: 12, fontFamily: "monospace"
    }}>{word}</span>
  );
}

function TestTag({ test }) {
  return (
    <span style={{
      padding: "4px 10px", borderRadius: 6,
      background: "rgba(255,255,255,0.06)",
      border: "1px solid rgba(255,255,255,0.1)",
      color: "#e2e8f0", fontSize: 12,
      display: "inline-flex", alignItems: "center", gap: 5
    }}>
      <span style={{ color: "#6c63ff", fontSize: 10 }}>🧪</span>
      {test}
    </span>
  );
}

export default function App() {
  const [symptoms, setSymptoms] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);
  const textareaRef = useRef(null);
  const resultsRef = useRef(null);

  async function analyze() {
    if (!symptoms.trim() || symptoms.trim().length < 10) {
      setError("Please describe your symptoms in more detail (at least 10 characters).");
      return;
    }
    setError("");
    setLoading(true);
    setResults(null);

    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          system: SYSTEM_PROMPT,
          messages: [{ role: "user", content: symptoms }],
        }),
      });
      const data = await res.json();
      const raw = data.content.map(b => b.text || "").join("");
      const cleaned = raw.replace(/```json|```/g, "").trim();
      const parsed = JSON.parse(cleaned);
      setResults(parsed);
      setHistory(h => [{ symptoms: symptoms.slice(0, 60) + "…", time: new Date().toLocaleTimeString() }, ...h].slice(0, 5));
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (e) {
      setError("Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  function useSample(sample) {
    setSymptoms(sample);
    setResults(null);
    textareaRef.current?.focus();
  }

  return (
    <div style={{
      minHeight: "100vh",
      background: "#080c14",
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
      color: "#e2e8f0",
      padding: "0 0 60px 0",
    }}>
      {/* Header */}
      <div style={{
        background: "linear-gradient(135deg, #0d1226 0%, #111827 50%, #0f0c29 100%)",
        borderBottom: "1px solid rgba(108,99,255,0.2)",
        padding: "28px 24px 24px",
        textAlign: "center",
        position: "relative",
        overflow: "hidden",
      }}>
        {/* Decorative circles */}
        <div style={{
          position: "absolute", top: -60, left: "20%", width: 200, height: 200,
          borderRadius: "50%", background: "rgba(108,99,255,0.06)",
          filter: "blur(40px)", pointerEvents: "none"
        }} />
        <div style={{
          position: "absolute", top: -40, right: "15%", width: 160, height: 160,
          borderRadius: "50%", background: "rgba(167,139,250,0.05)",
          filter: "blur(30px)", pointerEvents: "none"
        }} />
        <div style={{
          display: "inline-flex", alignItems: "center", gap: 8, marginBottom: 12,
          padding: "4px 14px", borderRadius: 20,
          background: "rgba(108,99,255,0.15)",
          border: "1px solid rgba(108,99,255,0.3)",
          fontSize: 12, color: "#a78bfa", letterSpacing: "0.05em"
        }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#6c63ff", display: "inline-block" }} />
          MEDICAL TEST RECOMMENDER
        </div>
        <h1 style={{
          fontSize: "clamp(22px, 4vw, 34px)", fontWeight: 700, margin: "0 0 8px",
          background: "linear-gradient(135deg, #ffffff 0%, #a78bfa 60%, #6c63ff 100%)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          letterSpacing: "-0.02em", lineHeight: 1.2
        }}>
          Symptom-Based Medical Test Recommender
        </h1>
        <p style={{ color: "#64748b", fontSize: 14, margin: 0, maxWidth: 480, marginInline: "auto" }}>
          NLP pipeline using TF-IDF + Word2Vec embeddings with cosine similarity ranking
        </p>

        {/* Tech badges */}
        <div style={{ display: "flex", justifyContent: "center", gap: 8, marginTop: 16, flexWrap: "wrap" }}>
          {["TF-IDF", "Word2Vec", "Cosine Similarity", "NLTK", "scikit-learn", "gensim"].map(b => (
            <span key={b} style={{
              padding: "2px 10px", borderRadius: 4, fontSize: 11,
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.1)",
              color: "#94a3b8", fontFamily: "monospace"
            }}>{b}</span>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: 860, margin: "0 auto", padding: "28px 16px 0" }}>
        {/* Input panel */}
        <div style={{
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 16, padding: 24, marginBottom: 20
        }}>
          <label style={{ fontSize: 13, color: "#94a3b8", display: "block", marginBottom: 10, fontWeight: 500 }}>
            DESCRIBE YOUR SYMPTOMS
          </label>
          <textarea
            ref={textareaRef}
            value={symptoms}
            onChange={e => setSymptoms(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) analyze(); }}
            placeholder="e.g. I've been feeling very tired, my skin looks pale, and I get breathless when climbing stairs..."
            rows={4}
            style={{
              width: "100%", background: "rgba(0,0,0,0.3)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 10, padding: "12px 14px",
              color: "#e2e8f0", fontSize: 15, lineHeight: 1.6,
              resize: "vertical", outline: "none", boxSizing: "border-box",
              fontFamily: "inherit",
            }}
          />
          {error && (
            <p style={{ color: "#ff4757", fontSize: 13, margin: "8px 0 0", display: "flex", alignItems: "center", gap: 6 }}>
              <span>⚠</span> {error}
            </p>
          )}

          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 14, flexWrap: "wrap", gap: 10 }}>
            <span style={{ fontSize: 12, color: "#475569" }}>
              {symptoms.length} chars · Ctrl+Enter to submit
            </span>
            <button
              onClick={analyze}
              disabled={loading}
              style={{
                padding: "10px 28px", borderRadius: 8, border: "none",
                background: loading ? "rgba(108,99,255,0.4)" : "linear-gradient(135deg, #6c63ff, #a78bfa)",
                color: "white", fontWeight: 600, fontSize: 14,
                cursor: loading ? "not-allowed" : "pointer",
                transition: "all 0.2s", letterSpacing: "0.02em",
                boxShadow: loading ? "none" : "0 4px 20px rgba(108,99,255,0.35)"
              }}
            >
              {loading ? "🔬 Analyzing…" : "🔍 Get Recommendations"}
            </button>
          </div>
        </div>

        {/* Sample symptoms */}
        <div style={{ marginBottom: 28 }}>
          <p style={{ fontSize: 12, color: "#475569", marginBottom: 10, letterSpacing: "0.05em" }}>
            TRY A SAMPLE:
          </p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {SAMPLE_SYMPTOMS.map((s, i) => (
              <button
                key={i}
                onClick={() => useSample(s)}
                style={{
                  padding: "6px 13px", borderRadius: 20, border: "1px solid rgba(255,255,255,0.1)",
                  background: "rgba(255,255,255,0.04)", color: "#94a3b8",
                  fontSize: 12, cursor: "pointer", textAlign: "left", transition: "all 0.15s",
                }}
                onMouseEnter={e => { e.target.style.borderColor = "rgba(108,99,255,0.5)"; e.target.style.color = "#a78bfa"; }}
                onMouseLeave={e => { e.target.style.borderColor = "rgba(255,255,255,0.1)"; e.target.style.color = "#94a3b8"; }}
              >
                {s.slice(0, 45)}…
              </button>
            ))}
          </div>
        </div>

        {/* Loading state */}
        {loading && (
          <div style={{
            background: "rgba(108,99,255,0.08)", border: "1px solid rgba(108,99,255,0.2)",
            borderRadius: 16, padding: 36, textAlign: "center"
          }}>
            <div style={{ marginBottom: 16 }}>
              {["🔤 Preprocessing text…", "📊 Computing TF-IDF vectors…", "🧠 Word2Vec embeddings…", "🎯 Ranking by cosine similarity…"].map((step, i) => (
                <div key={i} style={{
                  fontSize: 13, color: "#6c63ff", marginBottom: 6,
                  opacity: 0.4 + i * 0.2,
                }}>{step}</div>
              ))}
            </div>
            <div style={{ color: "#64748b", fontSize: 13 }}>Running NLP pipeline…</div>
          </div>
        )}

        {/* Results */}
        {results && (
          <div ref={resultsRef}>
            {/* Disclaimer */}
            <div style={{
              background: "rgba(255,165,2,0.08)", border: "1px solid rgba(255,165,2,0.2)",
              borderRadius: 10, padding: "10px 16px", marginBottom: 20,
              display: "flex", gap: 10, alignItems: "flex-start"
            }}>
              <span>⚠️</span>
              <span style={{ fontSize: 13, color: "#fbbf24" }}>{results.disclaimer}</span>
            </div>

            {/* Extracted keywords */}
            {results.symptom_keywords?.length > 0 && (
              <div style={{ marginBottom: 20 }}>
                <p style={{ fontSize: 12, color: "#475569", marginBottom: 8, letterSpacing: "0.05em" }}>
                  EXTRACTED SYMPTOM KEYWORDS
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {results.symptom_keywords.map((kw, i) => <KeywordChip key={i} word={kw} />)}
                </div>
              </div>
            )}

            {/* Priority tests callout */}
            {results.top_tests?.length > 0 && (
              <div style={{
                background: "linear-gradient(135deg, rgba(108,99,255,0.12), rgba(167,139,250,0.08))",
                border: "1px solid rgba(108,99,255,0.25)",
                borderRadius: 14, padding: "16px 20px", marginBottom: 24
              }}>
                <p style={{ fontSize: 12, color: "#a78bfa", marginBottom: 10, letterSpacing: "0.05em", fontWeight: 600 }}>
                  🏆 MOST COMMONLY RECOMMENDED TESTS
                </p>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {results.top_tests.map((t, i) => (
                    <span key={i} style={{
                      padding: "5px 12px", borderRadius: 8,
                      background: "rgba(108,99,255,0.2)",
                      border: "1px solid rgba(108,99,255,0.35)",
                      color: "#c4b5fd", fontSize: 13, fontWeight: 500
                    }}>
                      {i === 0 ? "①" : i === 1 ? "②" : "③"} {t}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendation cards */}
            <p style={{ fontSize: 12, color: "#475569", marginBottom: 14, letterSpacing: "0.05em" }}>
              TOP {results.recommendations?.length} RECOMMENDATIONS
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {results.recommendations?.map((rec, i) => {
                const urg = URGENCY_CONFIG[rec.urgency] || URGENCY_CONFIG.routine;
                return (
                  <div key={i} style={{
                    background: "rgba(255,255,255,0.03)",
                    border: `1px solid ${i === 0 ? "rgba(108,99,255,0.35)" : "rgba(255,255,255,0.07)"}`,
                    borderRadius: 14, padding: 20,
                    position: "relative", overflow: "hidden"
                  }}>
                    {i === 0 && (
                      <div style={{
                        position: "absolute", top: 0, left: 0, right: 0, height: 2,
                        background: "linear-gradient(90deg, #6c63ff, #a78bfa)"
                      }} />
                    )}
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <span style={{
                          width: 28, height: 28, borderRadius: "50%",
                          background: i === 0 ? "linear-gradient(135deg, #6c63ff, #a78bfa)" : "rgba(255,255,255,0.08)",
                          display: "flex", alignItems: "center", justifyContent: "center",
                          fontSize: 12, fontWeight: 700, color: "white", flexShrink: 0
                        }}>{rec.rank}</span>
                        <div>
                          <div style={{ fontWeight: 600, fontSize: 16, color: "#f1f5f9" }}>
                            {i === 0 ? <TypewriterText text={rec.condition} /> : rec.condition}
                          </div>
                          <div style={{ fontSize: 11, color: "#64748b", marginTop: 1 }}>
                            via {rec.model_used}
                          </div>
                        </div>
                      </div>
                      <span style={{
                        padding: "4px 10px", borderRadius: 6, fontSize: 11,
                        background: urg.bg, color: urg.color,
                        border: `1px solid ${urg.color}30`, fontWeight: 600
                      }}>
                        {urg.icon} {urg.label}
                      </span>
                    </div>

                    {/* Similarity score bar */}
                    <div style={{ marginBottom: 14 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span style={{ fontSize: 11, color: "#64748b" }}>Similarity Score</span>
                      </div>
                      <ScoreBar score={rec.similarity_score} />
                    </div>

                    {/* Reasoning */}
                    <p style={{ fontSize: 13, color: "#94a3b8", margin: "0 0 14px", lineHeight: 1.5 }}>
                      {rec.reasoning}
                    </p>

                    {/* Recommended tests */}
                    <div>
                      <p style={{ fontSize: 11, color: "#475569", marginBottom: 8, letterSpacing: "0.05em" }}>
                        RECOMMENDED TESTS
                      </p>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                        {rec.tests.map((t, j) => <TestTag key={j} test={t} />)}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* History */}
            {history.length > 1 && (
              <div style={{
                marginTop: 28, background: "rgba(255,255,255,0.02)",
                border: "1px solid rgba(255,255,255,0.06)",
                borderRadius: 12, padding: 16
              }}>
                <p style={{ fontSize: 12, color: "#475569", marginBottom: 12, letterSpacing: "0.05em" }}>
                  RECENT QUERIES
                </p>
                {history.map((h, i) => (
                  <div key={i} style={{
                    display: "flex", justifyContent: "space-between",
                    padding: "7px 0", borderBottom: i < history.length - 1 ? "1px solid rgba(255,255,255,0.05)" : "none",
                    fontSize: 13
                  }}>
                    <span style={{ color: "#64748b" }}>{h.symptoms}</span>
                    <span style={{ color: "#374151", fontSize: 11 }}>{h.time}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Footer */}
        <div style={{
          textAlign: "center", marginTop: 48, paddingTop: 24,
          borderTop: "1px solid rgba(255,255,255,0.05)"
        }}>
          <p style={{ fontSize: 12, color: "#374151" }}>
            Built with TF-IDF · Word2Vec · Cosine Similarity · NLTK · scikit-learn · gensim
          </p>
          <p style={{ fontSize: 11, color: "#1f2937", marginTop: 4 }}>
            ⚠️ Educational ML Portfolio Project — Not a substitute for medical advice
          </p>
        </div>
      </div>
    </div>
  );
}
