import type { ReplayTrace } from "../types";
import type { ApplicantStory } from "../lib/stories";
import { storyDecisionLabel, storyTone } from "../lib/stories";

interface ApplicantStoryViewProps {
  stories: ApplicantStory[];
  activeStoryIndex: number;
  trace: ReplayTrace | null;
  onSelectStory: (index: number) => void;
}

function coverageLabel(trace: ReplayTrace | null): string {
  if (!trace) {
    return "n/a";
  }
  const coverage = trace.episode.stats.decision_coverage;
  if (typeof coverage !== "number") {
    return "n/a";
  }
  return `${(coverage * 100).toFixed(1)}%`;
}

export function ApplicantStoryView(props: ApplicantStoryViewProps) {
  const { stories, activeStoryIndex, trace, onSelectStory } = props;
  const active = stories[activeStoryIndex] ?? null;

  return (
    <main className="story-layout">
      <section className="panel story-queue">
        <div className="panel-header">
          <h2>Applicant Story</h2>
          <p className="panel-subtext">{stories.length} applicants</p>
        </div>
        <div className="story-list">
          {stories.map((story, index) => (
            <button
              key={`${story.applicantIndex}-${story.startStep}-${story.endStep}`}
              className={`story-card ${index === activeStoryIndex ? "active" : ""}`}
              onClick={() => onSelectStory(index)}
            >
              <div className="story-top">
                <span className="story-id">Applicant #{story.applicantIndex + 1}</span>
                <span className={`story-badge ${storyTone(story)}`}>{storyDecisionLabel(story)}</span>
              </div>
              <p className="story-country">{story.countryName}</p>
              <div className="story-meta">
                <span>Inspects {story.inspectCount}</span>
                <span>Time {story.timeCost}</span>
                <span>Reward {story.rewardTotal.toFixed(2)}</span>
              </div>
              <p className="story-evidence">
                Revealed fields: <strong>{story.revealedFields.length}</strong>
              </p>
            </button>
          ))}
        </div>
      </section>

      <aside className="panel story-focus">
        <div className="panel-header">
          <h2>Focused Applicant</h2>
          <p className="panel-subtext">{active ? `Step ${active.startStep + 1}-${active.endStep + 1}` : "No data"}</p>
        </div>

        {!active ? (
          <div className="empty-block">Load a trace to render applicant stories.</div>
        ) : (
          <>
            <div className={`focus-decision ${storyTone(active)}`}>
              <p className="focus-title">{storyDecisionLabel(active)}</p>
              <p>
                {active.decision === "correct" && "Decision aligned with oracle legality."}
                {active.decision === "false_accept" && "Illegal applicant was accepted."}
                {active.decision === "false_reject" && "Legal applicant was rejected."}
                {active.decision === "pending" && "No decisive action was exported for this applicant."}
              </p>
            </div>

            <div className="focus-grid">
              <div className="focus-kpi">
                <span>Country</span>
                <strong>{active.countryName}</strong>
              </div>
              <div className="focus-kpi">
                <span>Inspections</span>
                <strong>{active.inspectCount}</strong>
              </div>
              <div className="focus-kpi">
                <span>Time Cost</span>
                <strong>{active.timeCost}</strong>
              </div>
              <div className="focus-kpi">
                <span>Reward</span>
                <strong>{active.rewardTotal.toFixed(2)}</strong>
              </div>
            </div>

            <div className="focus-block">
              <h3>Revealed During This Applicant</h3>
              {active.revealedFields.length === 0 ? (
                <p className="muted">No new field was revealed.</p>
              ) : (
                <div className="tag-row">
                  {active.revealedFields.map((field) => (
                    <span key={field} className="field-tag">
                      {field}
                    </span>
                  ))}
                </div>
              )}
            </div>

            <div className="focus-block">
              <h3>Rule Events</h3>
              {active.ruleUpdates.length === 0 ? (
                <p className="muted">No rule update event hit this applicant window.</p>
              ) : (
                <ul className="focus-list">
                  {active.ruleUpdates.map((ruleEvent) => (
                    <li key={ruleEvent}>{ruleEvent}</li>
                  ))}
                </ul>
              )}
            </div>

            <div className="focus-summary">
              <span>Total Reward</span>
              <strong>{trace?.episode.total_reward.toFixed(2) ?? "n/a"}</strong>
              <span>Decision Coverage</span>
              <strong>{coverageLabel(trace)}</strong>
            </div>
          </>
        )}
      </aside>
    </main>
  );
}
