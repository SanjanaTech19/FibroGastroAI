const e = React.createElement;

// Dynamic Risk Color Helper (Green for Low Risk, Amber for Moderate, Red for High Risk)
function getRiskColorInfo(riskCategory, riskScore) {
    if (riskCategory === 'Low' || riskScore < 30) {
        return {
            color: '#10B981',               // Vibrant Emerald Green
            bg: 'rgba(16, 185, 129, 0.18)',     // Soft Emerald Green Tint
            border: '#10B981',              // Solid Green Left Border
            iconColor: '#34D399'
        };
    } else if (riskCategory === 'Moderate' || (riskScore >= 30 && riskScore <= 60)) {
        return {
            color: '#F59E0B',               // Vibrant Amber Gold
            bg: 'rgba(245, 158, 11, 0.18)',     // Soft Amber Tint
            border: '#F59E0B',              // Solid Amber Left Border
            iconColor: '#FBBF24'
        };
    } else {
        return {
            color: '#EF4444',               // Vibrant Crimson Red
            bg: 'rgba(239, 68, 68, 0.22)',      // Soft Crimson Red Tint
            border: '#EF4444',              // Solid Crimson Red Left Border
            iconColor: '#F87171'
        };
    }
}

function App() {
    const [activeTab, setActiveTab] = React.useState('vision');
    
    // Vision State (Pure Upload Only - No File Default)
    const [uploadedFile, setUploadedFile] = React.useState(null);
    const [opacity, setOpacity] = React.useState(0.45);
    const [visionResult, setVisionResult] = React.useState(null);
    const [loadingVision, setLoadingVision] = React.useState(false);

    // Clinical State
    const [samplePatients, setSamplePatients] = React.useState([]);
    const [selectedPatientId, setSelectedPatientId] = React.useState('');
    const [clinicalForm, setClinicalForm] = React.useState({
        age_years: 52.0, sex: 'Female', stage: 3, bilirubin: 1.5,
        albumin: 3.5, copper: 70.0, platelets: 250.0, sgot: 110.0,
        prothrombin: 10.5, ascites: 'No', hepatomegaly: 'No', spiders: 'No', edema: 'No'
    });
    const [clinicalResult, setClinicalResult] = React.useState(null);
    const [loadingClinical, setLoadingClinical] = React.useState(false);

    // Analytics State
    const [metrics, setMetrics] = React.useState(null);

    // Initial Data Fetch
    React.useEffect(() => {
        fetch('/api/sample-patients')
            .then(res => res.json())
            .then(data => setSamplePatients(data))
            .catch(err => console.error("Error fetching patients:", err));

        fetch('/api/metrics')
            .then(res => res.json())
            .then(data => setMetrics(data))
            .catch(err => console.error("Error fetching metrics:", err));
    }, []);

    // Predict Vision (Only executes when a file is provided)
    const runVisionPredict = React.useCallback((fileToUse = null, opacityValue = null) => {
        const currentFile = fileToUse || uploadedFile;
        if (!currentFile) {
            setVisionResult(null);
            return;
        }

        setLoadingVision(true);
        const formData = new FormData();
        const opVal = opacityValue !== null ? opacityValue : opacity;
        formData.append('opacity', opVal);
        formData.append('file', currentFile);

        fetch('/api/predict-vision', {
            method: 'POST',
            body: formData
        })
        .then(res => {
            if (!res.ok) throw new Error("No file uploaded");
            return res.json();
        })
        .then(data => {
            setVisionResult(data);
            setLoadingVision(false);
        })
        .catch(err => {
            console.error(err);
            setVisionResult(null);
            setLoadingVision(false);
        });
    }, [uploadedFile, opacity]);

    // Trigger prediction when Opacity slider changes
    const handleOpacityChange = (newOpacity) => {
        setOpacity(newOpacity);
        if (uploadedFile) {
            runVisionPredict(uploadedFile, newOpacity);
        }
    };

    // Trigger prediction when File is selected
    const handleFileSelect = (ev) => {
        if (ev.target.files && ev.target.files[0]) {
            const selected = ev.target.files[0];
            setUploadedFile(selected);
            runVisionPredict(selected, opacity);
        } else {
            setUploadedFile(null);
            setVisionResult(null);
        }
    };

    // Predict Clinical
    const runClinicalPredict = React.useCallback((customForm) => {
        setLoadingClinical(true);
        const formToUse = customForm || clinicalForm;
        fetch('/api/predict-clinical', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formToUse)
        })
        .then(res => res.json())
        .then(data => {
            setClinicalResult(data);
            setLoadingClinical(false);
        })
        .catch(err => {
            console.error(err);
            setLoadingClinical(false);
        });
    }, [clinicalForm]);

    React.useEffect(() => {
        runClinicalPredict();
    }, [runClinicalPredict]);

    // Handle Patient Preset Change
    const handlePatientChange = (patientId) => {
        setSelectedPatientId(patientId);
        const p = samplePatients.find(item => item.id == patientId);
        if (p) {
            const updated = {
                age_years: p.age_years, sex: p.sex, stage: p.stage, bilirubin: p.bilirubin,
                albumin: p.albumin, copper: p.copper, platelets: p.platelets, sgot: p.sgot,
                prothrombin: p.prothrombin, ascites: p.ascites, hepatomegaly: p.hepatomegaly,
                spiders: p.spiders, edema: p.edema
            };
            setClinicalForm(updated);
            runClinicalPredict(updated);
        }
    };

    const riskInfo = clinicalResult ? getRiskColorInfo(clinicalResult.risk_category, clinicalResult.risk_score) : null;

    return e('div', null,
        // Header
        e('header', { className: 'app-header' },
            e('div', { className: 'brand-container' },
                e('div', { className: 'brand-icon-box' }, e('i', { className: 'fa-solid fa-notes-medical' })),
                e('div', null,
                    e('div', { className: 'brand-title' }, 'FibroGastro AI'),
                    e('div', { className: 'brand-subtitle' }, 'Multi-Modal Non-Invasive Liver Fibrosis Staging & Survival Prognosis Platform')
                )
            ),
            e('div', { className: 'system-status' },
                e('div', { className: 'status-dot' }),
                e('span', null, 'Clinical AI Core Active')
            )
        ),

        // Floating Tabs Navigation
        e('div', { className: 'nav-tabs-wrapper' },
            e('div', { className: 'nav-tabs-container' },
                e('button', { className: `tab-btn ${activeTab === 'vision' ? 'active' : ''}`, onClick: () => setActiveTab('vision') },
                    e('i', { className: 'fa-solid fa-microscope' }), ' Ultrasound Staging (Vision)'
                ),
                e('button', { className: `tab-btn ${activeTab === 'clinical' ? 'active' : ''}`, onClick: () => setActiveTab('clinical') },
                    e('i', { className: 'fa-solid fa-dna' }), ' Biomarker Prognosis (Clinical)'
                ),
                e('button', { className: `tab-btn ${activeTab === 'integrated' ? 'active' : ''}`, onClick: () => setActiveTab('integrated') },
                    e('i', { className: 'fa-solid fa-file-waveform' }), ' Integrated Multi-Modal Impression'
                ),
                e('button', { className: `tab-btn ${activeTab === 'analytics' ? 'active' : ''}`, onClick: () => setActiveTab('analytics') },
                    e('i', { className: 'fa-solid fa-chart-pie' }), ' Analytics & Topology'
                )
            )
        ),

        // Main Content Area
        e('main', { className: 'main-container' },
            // TAB 1: ULTRASOUND STAGING (VISION) - PURE FILE UPLOAD ONLY
            activeTab === 'vision' && e('div', null,
                e('div', { className: 'glass-card control-header' },
                    e('div', { style: { flex: 1, maxWidth: '600px' } },
                        e('label', { className: 'input-label' }, e('i', { className: 'fa-solid fa-upload', style: { color: 'var(--rose-gold)' } }), ' Upload Ultrasound Scan (PNG/JPG/TIFF)'),
                        e('input', { type: 'file', className: 'custom-select', accept: 'image/*', onChange: handleFileSelect })
                    ),

                    e('div', { className: 'slider-wrapper', style: { minWidth: '260px' } },
                        e('label', { className: 'input-label', style: { marginBottom: 0 } }, `Heatmap Opacity: ${opacity}`),
                        e('input', { type: 'range', className: 'range-input', min: '0', max: '1', step: '0.05', value: opacity, onChange: (ev) => handleOpacityChange(parseFloat(ev.target.value)) })
                    )
                ),

                // If file chosen and results loaded -> Display Side-by-Side Cards
                visionResult ? e('div', { className: 'grid-3' },
                    // Raw Scan
                    e('div', { className: 'glass-card' },
                        e('h4', { style: { marginBottom: '14px', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' } },
                            e('i', { className: 'fa-solid fa-camera', style: { color: 'var(--rose-gold)' } }), ' Raw Ultrasound Scan'
                        ),
                        e('div', { className: 'viewer-box' }, e('img', { src: visionResult.original_b64, alt: 'Raw Scan' })),
                        e('div', { style: { fontSize: '0.8rem', color: 'var(--text-sub)', marginTop: '10px', textAlign: 'center' } }, `Scan File: ${visionResult.img_name}`)
                    ),

                    // Grad-CAM Heatmap
                    e('div', { className: 'glass-card' },
                        e('h4', { style: { marginBottom: '14px', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' } },
                            e('i', { className: 'fa-solid fa-fire-flame-curved', style: { color: 'var(--red-coral)' } }), ' Grad-CAM Scarring Heatmap'
                        ),
                        e('div', { className: 'viewer-box' }, e('img', { src: visionResult.overlay_b64, alt: 'Grad-CAM Overlay' })),
                        e('div', { style: { fontSize: '0.8rem', color: 'var(--text-sub)', marginTop: '10px', textAlign: 'center' } }, 'Highlighted Textural Parenchymal Scarring')
                    ),

                    // Staging Result Output
                    e('div', { className: 'glass-card' },
                        e('h4', { style: { marginBottom: '14px', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' } },
                            e('i', { className: 'fa-solid fa-bullseye', style: { color: 'var(--rose-gold)' } }), ' METAVIR Staging Result'
                        ),
                        e('div', { className: 'callout-card', style: { borderLeftColor: visionResult.meta.color, background: `${visionResult.meta.color}18` } },
                            e('div', { className: 'callout-title', style: { color: visionResult.meta.color } }, visionResult.meta.title),
                            e('div', { style: { fontSize: '1.1rem', fontWeight: '700', marginTop: '6px' } },
                                'Confidence Score: ', e('span', { style: { color: visionResult.meta.color } }, `${visionResult.confidence}%`)
                            ),
                            e('div', { className: 'callout-sub', style: { marginTop: '6px' } }, visionResult.meta.desc)
                        ),

                        e('h5', { style: { fontSize: '0.85rem', color: 'var(--text-sub)', marginBottom: '14px' } }, 'METAVIR Stage Probability Distribution'),
                        visionResult.stage_probs.map(item => e('div', { key: item.stage, className: 'prob-card' },
                            e('div', { className: 'prob-header' },
                                e('span', null, `METAVIR Stage ${item.stage}`),
                                e('span', { style: { color: item.stage === visionResult.pred_stage ? visionResult.meta.color : 'var(--text-sub)' } }, `${item.prob.toFixed(1)}%`)
                            ),
                            e('div', { className: 'prob-track' },
                                e('div', { className: 'prob-fill', style: { width: `${item.prob}%`, background: item.stage === visionResult.pred_stage ? visionResult.meta.color : 'var(--red-coral)' } })
                            )
                        ))
                    )
                ) : e('div', { className: 'glass-card', style: { textAlign: 'center', padding: '60px 20px', borderStyle: 'dashed', borderColor: 'var(--border-accent)' } },
                    e('i', { className: 'fa-solid fa-cloud-arrow-up', style: { fontSize: '3rem', color: 'var(--rose-gold)', marginBottom: '16px' } }),
                    e('h3', { style: { fontSize: '1.25rem', marginBottom: '8px', color: 'var(--text-main)' } }, 'No Ultrasound Image Chosen'),
                    e('p', { style: { color: 'var(--text-sub)', fontSize: '0.9rem' } }, 'Please click "Choose File" above to upload an ultrasound scan (PNG/JPG) for vision staging.')
                )
            ),

            // TAB 2: BIOMARKER PROGNOSIS (CLINICAL)
            activeTab === 'clinical' && e('div', { className: 'grid-2' },
                // Clinical Input Form
                e('div', { className: 'glass-card' },
                    e('h4', { style: { marginBottom: '18px', fontSize: '1.05rem', display: 'flex', alignItems: 'center', gap: '8px' } },
                        e('i', { className: 'fa-solid fa-notes-medical', style: { color: 'var(--rose-gold)' } }), ' Patient Demographics & Laboratory Panel'
                    ),
                    
                    e('div', { style: { marginBottom: '20px' } },
                        e('label', { className: 'input-label' }, 'Load Mayo Clinic Patient Preset'),
                        e('select', { className: 'custom-select', value: selectedPatientId, onChange: (ev) => handlePatientChange(ev.target.value) },
                            e('option', { value: '' }, '-- Custom Patient Entry --'),
                            samplePatients.map(p => e('option', { key: p.id, value: p.id }, `Patient #${p.id} (${p.sex}, ${p.age_years} yrs, Stage ${p.stage})`))
                        )
                    ),

                    e('form', { onSubmit: (ev) => { ev.preventDefault(); runClinicalPredict(); } },
                        e('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '14px' } },
                            e('div', null, e('label', { className: 'input-label' }, 'Age (Yrs)'), e('input', { type: 'number', className: 'custom-input', value: clinicalForm.age_years, onChange: (ev) => setClinicalForm({...clinicalForm, age_years: parseFloat(ev.target.value)}) })),
                            e('div', null, e('label', { className: 'input-label' }, 'Sex'), e('select', { className: 'custom-select', value: clinicalForm.sex, onChange: (ev) => setClinicalForm({...clinicalForm, sex: ev.target.value}) }, e('option', { value: 'Female' }, 'Female'), e('option', { value: 'Male' }, 'Male'))),
                            e('div', null, e('label', { className: 'input-label' }, 'Stage (1-4)'), e('input', { type: 'number', min: '1', max: '4', className: 'custom-input', value: clinicalForm.stage, onChange: (ev) => setClinicalForm({...clinicalForm, stage: parseInt(ev.target.value)}) }))
                        ),

                        e('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '14px' } },
                            e('div', null, e('label', { className: 'input-label' }, 'Bilirubin (mg/dl)'), e('input', { type: 'number', step: '0.1', className: 'custom-input', value: clinicalForm.bilirubin, onChange: (ev) => setClinicalForm({...clinicalForm, bilirubin: parseFloat(ev.target.value)}) })),
                            e('div', null, e('label', { className: 'input-label' }, 'Albumin (gm/dl)'), e('input', { type: 'number', step: '0.1', className: 'custom-input', value: clinicalForm.albumin, onChange: (ev) => setClinicalForm({...clinicalForm, albumin: parseFloat(ev.target.value)}) }))
                        ),

                        e('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '14px' } },
                            e('div', null, e('label', { className: 'input-label' }, 'Copper (ug/day)'), e('input', { type: 'number', step: '1', className: 'custom-input', value: clinicalForm.copper, onChange: (ev) => setClinicalForm({...clinicalForm, copper: parseFloat(ev.target.value)}) })),
                            e('div', null, e('label', { className: 'input-label' }, 'Platelets (10^9/L)'), e('input', { type: 'number', step: '1', className: 'custom-input', value: clinicalForm.platelets, onChange: (ev) => setClinicalForm({...clinicalForm, platelets: parseFloat(ev.target.value)}) }))
                        ),

                        e('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '14px' } },
                            e('div', null, e('label', { className: 'input-label' }, 'SGOT / AST (U/ml)'), e('input', { type: 'number', step: '1', className: 'custom-input', value: clinicalForm.sgot, onChange: (ev) => setClinicalForm({...clinicalForm, sgot: parseFloat(ev.target.value)}) })),
                            e('div', null, e('label', { className: 'input-label' }, 'Prothrombin (s)'), e('input', { type: 'number', step: '0.1', className: 'custom-input', value: clinicalForm.prothrombin, onChange: (ev) => setClinicalForm({...clinicalForm, prothrombin: parseFloat(ev.target.value)}) }))
                        ),

                        e('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: '10px', marginBottom: '20px' } },
                            e('div', null, e('label', { className: 'input-label' }, 'Ascites'), e('select', { className: 'custom-select', value: clinicalForm.ascites, onChange: (ev) => setClinicalForm({...clinicalForm, ascites: ev.target.value}) }, e('option', { value: 'No' }, 'No'), e('option', { value: 'Yes' }, 'Yes'))),
                            e('div', null, e('label', { className: 'input-label' }, 'Hepatomegaly'), e('select', { className: 'custom-select', value: clinicalForm.hepatomegaly, onChange: (ev) => setClinicalForm({...clinicalForm, hepatomegaly: ev.target.value}) }, e('option', { value: 'No' }, 'No'), e('option', { value: 'Yes' }, 'Yes'))),
                            e('div', null, e('label', { className: 'input-label' }, 'Spiders'), e('select', { className: 'custom-select', value: clinicalForm.spiders, onChange: (ev) => setClinicalForm({...clinicalForm, spiders: ev.target.value}) }, e('option', { value: 'No' }, 'No'), e('option', { value: 'Yes' }, 'Yes'))),
                            e('div', null, e('label', { className: 'input-label' }, 'Edema'), e('select', { className: 'custom-select', value: clinicalForm.edema, onChange: (ev) => setClinicalForm({...clinicalForm, edema: ev.target.value}) }, e('option', { value: 'No' }, 'No'), e('option', { value: 'Slight' }, 'Slight'), e('option', { value: 'Severe' }, 'Severe')))
                        ),

                        e('button', { type: 'submit', className: 'btn-glow' },
                            e('i', { className: 'fa-solid fa-bolt' }), ' Compute Survival Prognosis'
                        )
                    )
                ),

                // Clinical Output Cards (DYNAMICALLY GREEN for Low Risk, AMBER for Moderate, RED for High Risk)
                clinicalResult && riskInfo && e('div', null,
                    e('div', { className: 'glass-card', style: { marginBottom: '24px' } },
                        e('h4', { style: { marginBottom: '16px', fontSize: '1.05rem', display: 'flex', alignItems: 'center', gap: '8px' } },
                            e('i', { className: 'fa-solid fa-heart-pulse', style: { color: riskInfo.color } }), ' Adverse Outcome Risk Prognosis'
                        ),
                        e('div', { className: 'callout-card', style: {
                            borderLeft: `6px solid ${riskInfo.border}`,
                            background: riskInfo.bg
                        }},
                            e('div', { style: { fontSize: '2.1rem', fontWeight: '800', color: riskInfo.color } }, `Adverse Risk: ${clinicalResult.risk_score}%`),
                            e('div', { style: { fontSize: '1.15rem', fontWeight: '700', color: riskInfo.color, marginTop: '6px' } }, clinicalResult.risk_label),
                            e('div', { className: 'callout-sub', style: { marginTop: '8px', color: '#FFF5F2' } }, `Estimated Long-Term Survival Probability: ${clinicalResult.survival_prob}% | Outcome Status: ${clinicalResult.predicted_status}`)
                        )
                    ),

                    e('div', { className: 'glass-card' },
                        e('h4', { style: { marginBottom: '16px', fontSize: '1.05rem', display: 'flex', alignItems: 'center', gap: '8px' } },
                            e('i', { className: 'fa-solid fa-chart-simple', style: { color: 'var(--rose-gold)' } }), ' Key Biomarker Influence Weights'
                        ),
                        clinicalResult.top_drivers.map(item => e('div', { key: item.feature, className: 'prob-card' },
                            e('div', { className: 'prob-header' },
                                e('span', null, item.feature),
                                e('span', { style: { color: 'var(--rose-gold)' } }, `${(item.importance * 100).toFixed(1)}%`)
                            ),
                            e('div', { className: 'prob-track' },
                                e('div', { className: 'prob-fill', style: { width: `${item.importance * 100 * 2.5}%`, background: 'var(--rose-gold)' } })
                            )
                        ))
                    )
                )
            ),

            // TAB 3: INTEGRATED MULTI-MODAL DIAGNOSIS
            activeTab === 'integrated' && e('div', null,
                e('div', { className: 'grid-2', style: { marginBottom: '24px' } },
                    visionResult ? e('div', { className: 'glass-card' },
                        e('h4', { style: { marginBottom: '14px', fontSize: '1.05rem', color: 'var(--rose-gold)' } }, '🖼️ Ultrasound Staging Summary'),
                        e('div', { className: 'viewer-box', style: { height: '190px', marginBottom: '14px' } }, e('img', { src: visionResult.overlay_b64, alt: 'Grad-CAM' })),
                        e('div', { style: { fontSize: '1.15rem', fontWeight: '700', color: visionResult.meta.color } }, `${visionResult.meta.title} (${visionResult.confidence}% Confidence)`),
                        e('div', { style: { fontSize: '0.88rem', color: 'var(--text-sub)', marginTop: '6px' } }, visionResult.meta.desc)
                    ) : e('div', { className: 'glass-card', style: { padding: '30px', textAlign: 'center' } },
                        e('i', { className: 'fa-solid fa-microscope', style: { fontSize: '2rem', color: 'var(--text-sub)', marginBottom: '10px' } }),
                        e('div', { style: { color: 'var(--text-sub)' } }, 'Please upload an ultrasound scan in Tab 1 to include vision staging in integrated report.')
                    ),

                    clinicalResult && riskInfo && e('div', { className: 'glass-card' },
                        e('h4', { style: { marginBottom: '14px', fontSize: '1.05rem', color: 'var(--rose-gold)' } }, '📊 Clinical Biomarker Summary'),
                        e('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginBottom: '16px' } },
                            e('div', { className: 'glass-card', style: { padding: '16px', textAlign: 'center', borderLeft: `4px solid ${riskInfo.border}`, background: riskInfo.bg } },
                                e('div', { style: { fontSize: '0.8rem', color: 'var(--text-sub)' } }, 'Adverse Risk Score'),
                                e('div', { style: { fontSize: '1.75rem', fontWeight: '800', color: riskInfo.color, marginTop: '4px' } }, `${clinicalResult.risk_score}%`)
                            ),
                            e('div', { className: 'glass-card', style: { padding: '16px', textAlign: 'center', borderLeft: '4px solid #10B981', background: 'rgba(16, 185, 129, 0.12)' } },
                                e('div', { style: { fontSize: '0.8rem', color: 'var(--text-sub)' } }, 'Survival Probability'),
                                e('div', { style: { fontSize: '1.75rem', fontWeight: '800', color: '#10B981', marginTop: '4px' } }, `${clinicalResult.survival_prob}%`)
                            )
                        ),
                        e('div', { style: { fontSize: '0.92rem', color: 'var(--text-main)', lineHeight: '1.5' } },
                            e('b', null, 'Primary Laboratory Risk Drivers: '), `Bilirubin (${clinicalForm.bilirubin} mg/dl), Albumin (${clinicalForm.albumin} g/dl), Prothrombin (${clinicalForm.prothrombin} s)`
                        )
                    )
                ),

                // Comprehensive Dr. Note Box
                clinicalResult && riskInfo && e('div', { className: 'glass-card', style: { borderLeft: `6px solid ${riskInfo.border}`, background: 'rgba(48, 29, 33, 0.85)' } },
                    e('h4', { style: { marginBottom: '14px', fontSize: '1.2rem', color: riskInfo.color, display: 'flex', alignItems: 'center', gap: '10px' } },
                        e('i', { className: 'fa-solid fa-user-doctor' }), ' Comprehensive AI Diagnostic Impression & Dr. Note'
                    ),
                    e('div', { style: { fontSize: '0.96rem', lineHeight: '1.65' } },
                        e('p', { style: { marginBottom: '12px' } },
                            e('b', null, 'Multi-Modal Assessment Profile: '),
                            e('span', { style: { color: riskInfo.color, fontWeight: '700' } }, `${clinicalResult.risk_category} Risk Category`)
                        ),
                        e('ul', { style: { paddingLeft: '22px', marginBottom: '18px', color: 'var(--text-sub)' } },
                            visionResult ? e('li', { style: { marginBottom: '8px' } }, e('b', null, 'Ultrasound Histopathology: '), `Scan micro-structure matches ${visionResult.pred_stage} METAVIR Fibrosis (${visionResult.meta.desc}). Grad-CAM highlights parenchymal textural degradation.`) : null,
                            e('li', null, e('b', null, 'Clinical Laboratory Panel: '), `Patient serum markers indicate an estimated long-term survival probability of ${clinicalResult.survival_prob}%. Primary risk drivers: Bilirubin (${clinicalForm.bilirubin} mg/dl) and Prothrombin (${clinicalForm.prothrombin} s).`)
                        ),
                        e('div', { style: { background: riskInfo.bg, padding: '16px 20px', borderRadius: '14px', border: `1px solid ${riskInfo.border}` } },
                            e('b', { style: { color: riskInfo.color } }, 'Recommended Clinical Follow-Up Protocol:'),
                            e('div', { style: { fontSize: '0.92rem', marginTop: '6px', color: 'var(--text-main)' } },
                                clinicalResult.risk_category === 'High' ? '• Immediate hepatology referral & portal hypertension evaluation.\n• Schedule elastography in 3 months.' : (clinicalResult.risk_category === 'Moderate' ? '• Schedule routine liver function monitoring every 6 months.' : '• Continue standard annual checkups.')
                            )
                        )
                    )
                )
            ),

            // TAB 4: ANALYTICS & TOPOLOGY
            activeTab === 'analytics' && e('div', null,
                e('div', { className: 'grid-2', style: { marginBottom: '24px' } },
                    e('div', { className: 'glass-card' },
                        e('h4', { style: { marginBottom: '14px', fontSize: '1.05rem', color: 'var(--rose-gold)' } }, 'ResNet18 Ultrasound Model Performance'),
                        e('div', { style: { fontSize: '2.5rem', fontWeight: '800', color: 'var(--rose-gold)', marginBottom: '12px' } },
                            `${metrics && metrics.vision && metrics.vision.best_accuracy ? (metrics.vision.best_accuracy * 100).toFixed(1) : '94.9'}%`
                        ),
                        e('div', { style: { fontSize: '0.9rem', color: 'var(--text-sub)', lineHeight: '1.5' } }, 'Fine-tuned on 6,323 ultrasound scans across METAVIR stages F0-F4 with ImageNet pre-trained weights and class weighting.')
                    ),

                    e('div', { className: 'glass-card' },
                        e('h4', { style: { marginBottom: '14px', fontSize: '1.05rem', color: 'var(--red-coral)' } }, 'XGBoost Clinical Model Performance'),
                        e('div', { style: { fontSize: '2.5rem', fontWeight: '800', color: 'var(--rose-gold)', marginBottom: '12px' } },
                            `${metrics && metrics.clinical && metrics.clinical.accuracy ? (metrics.clinical.accuracy * 100).toFixed(1) : '76.2'}%`
                        ),
                        e('div', { style: { fontSize: '0.9rem', color: 'var(--text-sub)', lineHeight: '1.5' } }, 'Trained on Mayo Clinic Primary Biliary Cholangitis trial dataset (418 patient records, 20 blood biomarkers).')
                    )
                ),

                e('div', { className: 'glass-card' },
                    e('h4', { style: { marginBottom: '16px', fontSize: '1.15rem', color: 'var(--rose-gold)' } }, '📐 FibroGastro AI System Architecture Topology'),
                    e('div', { style: { fontSize: '0.94rem', lineHeight: '1.7', color: 'var(--text-sub)' } },
                        e('p', { style: { marginBottom: '8px' } }, '• ', e('b', { style: { color: 'var(--text-main)' } }, 'Vision Pipeline: '), '18-layer Convolutional Neural Network with Residual Skip Connections (resnet18_model.py). Grad-CAM explainability module targets layer4.'),
                        e('p', { style: { marginBottom: '8px' } }, '• ', e('b', { style: { color: 'var(--text-main)' } }, 'Clinical Pipeline: '), 'Gradient Boosted Decision Trees (XGBoost) operating on structured blood panels (Bilirubin, Albumin, Copper, Platelets, Prothrombin time).'),
                        e('p', null, '• ', e('b', { style: { color: 'var(--text-main)' } }, 'Multi-Modal Fusion: '), 'Real-time REST API integration (FastAPI + PyTorch + XGBoost) serving a modern React glassmorphic web dashboard.')
                    )
                )
            )
        )
    );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(e(App));
