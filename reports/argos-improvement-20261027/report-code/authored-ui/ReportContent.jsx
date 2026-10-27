import React from 'react';
import { ReportSection, RichNarrative, DataComponent, EvidenceChart, SortableRegion, SortableItem, useDataApp } from '../../data-app-public.jsx';
import './argos-report.css';
export function ReportContent() {
  const { snapshot, appTitle, visible } = useDataApp();
  const sections = snapshot.report?.sections ?? [];
  return <article className="report-page argos-report">
    <header className="report-header">
      <RichNarrative id="argos-title" value={`# ${appTitle}`} />
      <RichNarrative id="argos-deck" className="report-deck" value={snapshot.report?.introduction ?? ''} />
    </header>
    <SortableRegion id="argos-sections" label="실험 보고서" variant="stack" authoredOrder={sections.map(s=>s.id)} className="report-sortable-sections">
      {sections.filter(s=>visible(s.id)).map(s=>{
        const rows = snapshot.queries[s.queryId]?.rows ?? [];
        return <SortableItem key={s.id} id={s.id} label={s.title} kind="narrative">
          <ReportSection id={s.id} queryId={s.queryId} title={s.title} sourceRows={rows} showHeading={false}>
            <RichNarrative id={`${s.id}:body`} value={`## ${s.title}\n\n${s.body}`} />
            {(s.charts??[]).map(c=><EvidenceChart key={c.id} id={c.id} queryId={s.queryId} title={c.title} spec={c.spec} rows={rows} sourceRows={rows} height={440} />)}
            {s.columns?.length>0 && <DataComponent id={`${s.id}:table`} queryId={s.queryId} title={s.title} kind="table" sourceRows={rows}>
              <div className="argos-table-scroll" data-reviewed-rows>
                <table className="argos-table"><thead><tr>{s.columns.map(c=><th key={c.field} className={c.numeric?'numeric':''}>{c.label}</th>)}</tr></thead>
                  <tbody>{rows.map((r,i)=><tr key={r.id??i}>{s.columns.map(c=>{
                    const value=r[c.field]===null || r[c.field]===undefined?'미실행':String(r[c.field]);
                    return <td key={c.field} className={c.numeric?'numeric':''}>{value.includes('](https://')?<RichNarrative id={`${s.id}:row-${i}:${c.field}`} className="argos-cell-narrative" value={value}/>:value}</td>;
                  })}</tr>)}</tbody>
                </table>
              </div>
            </DataComponent>}
          </ReportSection>
        </SortableItem>;
      })}
    </SortableRegion>
  </article>;
}
