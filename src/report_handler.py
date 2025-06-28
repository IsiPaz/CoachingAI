import time
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json


class ReportHandler:
    """
    Handles collection and generation of coaching session reports.
    Analyzes comprehensive metrics and generates AI-powered feedback.
    """
    
    def __init__(self, 
                 openai_client=None,
                 sample_interval: float = 2.0):
        """
        Initialize the Report handler.
        
        Args:
            openai_client: OpenAI client for ChatGPT report generation
            sample_interval: Interval in seconds between metric samples
        """
        self.openai_client = openai_client
        self.sample_interval = sample_interval
        
        # Session tracking
        self.session_start_time = None
        self.last_sample_time = 0
        
        # Metrics storage
        self.session_metrics = []
        
        # Analysis cache
        self._analysis_cache = None
        
        print(f"ReportHandler initialized (sample interval: {sample_interval}s)")
        if openai_client:
            print("ChatGPT report generation enabled")
    
    def start_session(self) -> None:
        """Start a new coaching session."""
        self.session_start_time = time.time()
        self.last_sample_time = 0
        self.session_metrics = []
        self._analysis_cache = None
        print("📊 Coaching session started - collecting metrics...")
    
    def collect_metrics(self, 
                       emotion_info: Optional[Dict],
                       iris_info: Optional[Dict],
                       pose_info: Optional[Dict],
                       distance_info: Optional[Dict],
                       hand_info: Optional[Dict]) -> None:
        """
        Collect metrics from current frame if interval has passed.
        
        Args:
            emotion_info: Emotion recognition results
            iris_info: Iris tracking results
            pose_info: Pose analysis results
            distance_info: Distance measurement results
            hand_info: Hand tracking results
        """
        if self.session_start_time is None:
            return
            
        current_time = time.time()
        
        # Check if enough time has passed since last sample
        if current_time - self.last_sample_time < self.sample_interval:
            return
        
        # Create metrics point
        metrics_point = {
            'timestamp': current_time - self.session_start_time,
            'raw_timestamp': current_time
        }
        
        # Collect emotion metrics
        if emotion_info:
            metrics_point['emotion'] = {
                'predicted_emotion': emotion_info.get('predicted_emotion', 'unknown'),
                'confidence': emotion_info.get('confidence', 0),
                'valence': emotion_info.get('valence', 0),
                'arousal': emotion_info.get('arousal', 0)
            }
        
        # Collect posture metrics
        if pose_info and 'posture_metrics' in pose_info:
            posture = pose_info['posture_metrics']
            metrics_point['posture'] = {
                'overall_score': posture.get('overall_posture_score', 0),
                'trunk_fb': posture.get('trunk_inclination_fb', 0),
                'trunk_lr': posture.get('trunk_inclination_lr', 0),
                'shoulder_symmetry': posture.get('shoulder_symmetry_score', 0),
                'shoulder_status': posture.get('shoulder_symmetry_status', 'unknown'),
                'head_tilt': posture.get('head_tilt', 0),
                'head_turn': posture.get('head_turn', 0),
                'orientation': posture.get('orientation', 'unknown')
            }
        
        # Collect gaze/iris metrics
        if iris_info:
            iris_pos = iris_info.get('iris_position', {})
            metrics_point['gaze'] = {
                'centered': iris_pos.get('average_centering', 1.0) < 0.3,
                'centering_score': iris_pos.get('average_centering', 1.0),
                'horizontal_offset': iris_pos.get('average_horizontal_offset', 0),
                'vertical_offset': iris_pos.get('average_vertical_offset', 0),
                'eye_aperture': iris_info.get('average_eye_aperture', 0),
                'eyes_closed': iris_info.get('eyes_closed', False),
                'blink_count': iris_info.get('total_blinks', 0)
            }
        
        # Collect distance metrics
        if distance_info:
            metrics_point['distance'] = {
                'distance_cm': distance_info.get('distance_cm', 0),
                'status': distance_info.get('distance_status', 'unknown'),
                'quality': distance_info.get('distance_quality', 'unknown'),
                'face_percentage': distance_info.get('face_size_percentage', 0),
                'is_optimal': distance_info.get('distance_quality') == 'excellent'
            }
        
        # Collect hand metrics
        if hand_info:
            hand_metrics = {
                'hands_detected': hand_info.get('hands_detected', False),
                'num_hands': hand_info.get('num_hands', 0),
                'face_covered': hand_info.get('face_interference', {}).get('sustained_interference', False),
                'interference_score': hand_info.get('face_interference', {}).get('interference_score', 0),
                'hand_symmetry': hand_info.get('hand_symmetry', 0)
            }
            
            # Analyze individual hands
            if hand_info.get('hand_metrics'):
                dominant_gesture = 'none'
                avg_openness = 0
                gesture_activity = 0
                
                for hand_side, hand_data in hand_info['hand_metrics'].items():
                    if hand_data.get('is_dominant'):
                        dominant_gesture = hand_data.get('primary_gesture', 'none')
                    avg_openness += hand_data.get('openness', 0)
                    gesture_activity += hand_data.get('gesticulation_intensity', 0)
                
                num_hands = len(hand_info['hand_metrics'])
                hand_metrics.update({
                    'dominant_gesture': dominant_gesture,
                    'avg_openness': avg_openness / num_hands if num_hands > 0 else 0,
                    'gesture_activity': gesture_activity / num_hands if num_hands > 0 else 0
                })
            
            metrics_point['hands'] = hand_metrics
        
        # Store the metrics point
        self.session_metrics.append(metrics_point)
        self.last_sample_time = current_time
        
        # Clear analysis cache
        self._analysis_cache = None
    
    def analyze_session(self) -> Dict[str, Any]:
        """
        Analyze collected metrics for the session.
        
        Returns:
            Comprehensive analysis of the session
        """
        if self._analysis_cache is not None:
            return self._analysis_cache
        
        if not self.session_metrics:
            return {'error': 'No metrics collected'}
        
        analysis = {
            'session_info': {
                'duration_minutes': (time.time() - self.session_start_time) / 60 if self.session_start_time else 0,
                'total_samples': len(self.session_metrics),
                'sample_rate': f"Every {self.sample_interval}s"
            }
        }
        
        # Analyze emotions
        analysis['emotions'] = self._analyze_emotions()
        
        # Analyze posture
        analysis['posture'] = self._analyze_posture()
        
        # Analyze gaze
        analysis['gaze'] = self._analyze_gaze()
        
        # Analyze distance
        analysis['distance'] = self._analyze_distance()
        
        # Analyze hands
        analysis['hands'] = self._analyze_hands()
        
        # Overall performance score
        analysis['overall'] = self._calculate_overall_performance(analysis)
        
        # Cache the analysis
        self._analysis_cache = analysis
        
        return analysis
    
    def _analyze_emotions(self) -> Dict:
        """Analyze emotional patterns during the session."""
        emotions = [m['emotion'] for m in self.session_metrics if 'emotion' in m]
        if not emotions:
            return {'status': 'no_data'}
        
        # Emotion distribution
        emotion_counts = defaultdict(int)
        valences = []
        arousals = []
        confidences = []
        
        for emotion in emotions:
            emotion_counts[emotion['predicted_emotion']] += 1
            valences.append(emotion['valence'])
            arousals.append(emotion['arousal'])
            confidences.append(emotion['confidence'])
        
        # Most common emotion
        most_common = max(emotion_counts.items(), key=lambda x: x[1])
        
        return {
            'most_common_emotion': most_common[0],
            'emotion_percentage': (most_common[1] / len(emotions)) * 100,
            'avg_valence': np.mean(valences),
            'avg_arousal': np.mean(arousals),
            'avg_confidence': np.mean(confidences),
            'emotional_stability': np.std(valences),  # Lower is more stable
            'energy_level': 'high' if np.mean(arousals) > 0.2 else 'low' if np.mean(arousals) < -0.2 else 'moderate',
            'mood_quality': 'positive' if np.mean(valences) > 0.1 else 'negative' if np.mean(valences) < -0.1 else 'neutral',
            'distribution': dict(emotion_counts)
        }
    
    def _analyze_posture(self) -> Dict:
        """Analyze posture patterns during the session."""
        postures = [m['posture'] for m in self.session_metrics if 'posture' in m]
        if not postures:
            return {'status': 'no_data'}
        
        scores = [p['overall_score'] for p in postures]
        orientations = [p['orientation'] for p in postures]
        shoulder_statuses = [p['shoulder_status'] for p in postures]
        
        # Calculate percentages
        frontal_count = sum(1 for o in orientations if o.lower() == 'frontal')
        frontal_percentage = (frontal_count / len(orientations)) * 100
        
        good_shoulder_count = sum(1 for s in shoulder_statuses if s == 'Correct')
        shoulder_symmetry_percentage = (good_shoulder_count / len(shoulder_statuses)) * 100
        
        avg_score = np.mean(scores)
        
        return {
            'avg_posture_score': avg_score,
            'posture_quality': 'excellent' if avg_score > 0.7 else 'good' if avg_score > 0.5 else 'needs_improvement',
            'frontal_percentage': frontal_percentage,
            'shoulder_symmetry_percentage': shoulder_symmetry_percentage,
            'posture_consistency': 1.0 - np.std(scores),  # Higher is more consistent
            'best_score': np.max(scores),
            'worst_score': np.min(scores)
        }
    
    def _analyze_gaze(self) -> Dict:
        """Analyze gaze patterns during the session."""
        gazes = [m['gaze'] for m in self.session_metrics if 'gaze' in m]
        if not gazes:
            return {'status': 'no_data'}
        
        centered_count = sum(1 for g in gazes if g['centered'])
        centered_percentage = (centered_count / len(gazes)) * 100
        
        centering_scores = [g['centering_score'] for g in gazes]
        avg_centering = np.mean(centering_scores)
        
        eye_apertures = [g['eye_aperture'] for g in gazes]
        avg_aperture = np.mean(eye_apertures)
        
        eyes_closed_count = sum(1 for g in gazes if g.get('eyes_closed', False))
        eyes_closed_percentage = (eyes_closed_count / len(gazes)) * 100
        
        return {
            'gaze_centered_percentage': centered_percentage,
            'avg_centering_score': avg_centering,
            'gaze_quality': 'excellent' if centered_percentage > 80 else 'good' if centered_percentage > 60 else 'needs_improvement',
            'avg_eye_aperture': avg_aperture,
            'alertness_level': 'high' if avg_aperture > 0.25 else 'low' if avg_aperture < 0.15 else 'moderate',
            'eyes_closed_percentage': eyes_closed_percentage,
            'total_blinks': gazes[-1]['blink_count'] if gazes and 'blink_count' in gazes[-1] else 0
        }
    
    def _analyze_distance(self) -> Dict:
        """Analyze distance patterns during the session."""
        distances = [m['distance'] for m in self.session_metrics if 'distance' in m]
        if not distances:
            return {'status': 'no_data'}
        
        optimal_count = sum(1 for d in distances if d.get('is_optimal', False))
        optimal_percentage = (optimal_count / len(distances)) * 100
        
        avg_distance = np.mean([d['distance_cm'] for d in distances])
        
        # Status distribution
        status_counts = defaultdict(int)
        for d in distances:
            status_counts[d['status']] += 1
        
        return {
            'optimal_percentage': optimal_percentage,
            'avg_distance_cm': avg_distance,
            'distance_quality': 'excellent' if optimal_percentage > 80 else 'good' if optimal_percentage > 60 else 'needs_adjustment',
            'most_common_status': max(status_counts.items(), key=lambda x: x[1])[0],
            'status_distribution': dict(status_counts)
        }
    
    def _analyze_hands(self) -> Dict:
        """Analyze hand usage patterns during the session."""
        hands = [m['hands'] for m in self.session_metrics if 'hands' in m]
        if not hands:
            return {'status': 'no_data'}
        
        hands_visible_count = sum(1 for h in hands if h.get('hands_detected', False))
        hands_visible_percentage = (hands_visible_count / len(hands)) * 100
        
        face_covered_count = sum(1 for h in hands if h.get('face_covered', False))
        face_covered_percentage = (face_covered_count / len(hands)) * 100
        
        gesture_activities = [h.get('gesture_activity', 0) for h in hands if 'gesture_activity' in h]
        avg_gesture_activity = np.mean(gesture_activities) if gesture_activities else 0
        
        # Gesture distribution
        gestures = [h.get('dominant_gesture', 'none') for h in hands if 'dominant_gesture' in h]
        gesture_counts = defaultdict(int)
        for gesture in gestures:
            gesture_counts[gesture] += 1
        
        return {
            'hands_visible_percentage': hands_visible_percentage,
            'face_covered_percentage': face_covered_percentage,
            'hand_usage_quality': 'excellent' if hands_visible_percentage > 70 and face_covered_percentage < 10 else 'needs_improvement',
            'avg_gesture_activity': avg_gesture_activity,
            'gesture_engagement': 'high' if avg_gesture_activity > 0.3 else 'low' if avg_gesture_activity < 0.1 else 'moderate',
            'most_common_gesture': max(gesture_counts.items(), key=lambda x: x[1])[0] if gesture_counts else 'none',
            'gesture_distribution': dict(gesture_counts)
        }
    
    def _calculate_overall_performance(self, analysis: Dict) -> Dict:
        """Calculate overall performance metrics."""
        scores = []
        
        # Posture score (30%)
        if 'posture' in analysis and 'avg_posture_score' in analysis['posture']:
            posture_score = analysis['posture']['avg_posture_score']
            scores.append(('posture', posture_score, 0.3))
        
        # Gaze score (25%)
        if 'gaze' in analysis and 'gaze_centered_percentage' in analysis['gaze']:
            gaze_score = analysis['gaze']['gaze_centered_percentage'] / 100
            scores.append(('gaze', gaze_score, 0.25))
        
        # Distance score (20%)
        if 'distance' in analysis and 'optimal_percentage' in analysis['distance']:
            distance_score = analysis['distance']['optimal_percentage'] / 100
            scores.append(('distance', distance_score, 0.2))
        
        # Emotional score (15%)
        if 'emotions' in analysis and 'avg_valence' in analysis['emotions']:
            emotion_score = (analysis['emotions']['avg_valence'] + 1) / 2  # Normalize -1,1 to 0,1
            scores.append(('emotion', emotion_score, 0.15))
        
        # Hand usage score (10%)
        if 'hands' in analysis and 'hands_visible_percentage' in analysis['hands']:
            hand_score = analysis['hands']['hands_visible_percentage'] / 100
            # Penalize face covering
            if analysis['hands']['face_covered_percentage'] > 0:
                hand_score *= (1 - analysis['hands']['face_covered_percentage'] / 100)
            scores.append(('hands', hand_score, 0.1))
        
        # Calculate weighted average
        if scores:
            weighted_sum = sum(score * weight for _, score, weight in scores)
            total_weight = sum(weight for _, _, weight in scores)
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 0.5
        
        return {
            'overall_score': overall_score,
            'performance_level': 'excellent' if overall_score > 0.8 else 'good' if overall_score > 0.6 else 'needs_improvement',
            'component_scores': {name: score for name, score, _ in scores}
        }
    
    def generate_report(self, save_to_file: bool = True) -> str:
        """
        Generate comprehensive coaching report.
        
        Args:
            save_to_file: Whether to save report to a text file
            
        Returns:
            Report content as string
        """
        analysis = self.analyze_session()
        
        if 'error' in analysis:
            return f"Cannot generate report: {analysis['error']}"
        
        # Try ChatGPT report first, fallback to basic report
        if self.openai_client:
            try:
                report_content = self._generate_chatgpt_report(analysis)
            except Exception as e:
                print(f"ChatGPT report failed, using basic report: {e}")
                report_content = self._generate_basic_report(analysis)
        else:
            report_content = self._generate_basic_report(analysis)
        
        # Save to file if requested
        if save_to_file:
            timestamp = int(time.time())
            filename = f"coaching_report_{timestamp}.txt"
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                print(f"📊 Coaching report saved: {filename}")
            except Exception as e:
                print(f"Failed to save report: {e}")
        
        return report_content
    
    def _generate_chatgpt_report(self, analysis: Dict) -> str:
        """Generate report using ChatGPT analysis."""
        
        # Prepare context for ChatGPT
        context = self._prepare_analysis_context(analysis)
        
        # Generate ChatGPT analysis
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un coach profesional de comunicación virtual. Analiza estas métricas de una sesión de video coaching y proporciona un reporte constructivo, específico y actionable. Sé directo pero motivador."},
                {"role": "user", "content": f"""{context}

Genera un reporte de coaching que incluya:

1. RESUMEN EJECUTIVO (2-3 líneas sobre el desempeño general)
2. FORTALEZAS PRINCIPALES (máximo 3 puntos específicos)
3. ÁREAS DE MEJORA PRIORITARIAS (máximo 3 puntos con acciones concretas)
4. RECOMENDACIÓN CLAVE (1 acción específica más importante)

Usa un tono profesional pero cercano. Máximo 250 palabras. Sin formato markdown."""}
            ],
            max_tokens=400,
            temperature=0.7,
            timeout=15
        )
        
        chatgpt_analysis = response.choices[0].message.content.strip()
        
        # Combine with detailed metrics
        return self._format_final_report(chatgpt_analysis, analysis)
    
    def _generate_basic_report(self, analysis: Dict) -> str:
        """Generate basic report without ChatGPT."""
        
        # Analyze strengths and weaknesses
        strengths = []
        improvements = []
        
        # Posture analysis
        if analysis.get('posture', {}).get('avg_posture_score', 0) > 0.7:
            strengths.append("Excelente postura corporal y presencia física")
        elif analysis.get('posture', {}).get('avg_posture_score', 0) < 0.5:
            improvements.append("Mejorar alineación corporal y postura")
        
        # Gaze analysis
        if analysis.get('gaze', {}).get('gaze_centered_percentage', 0) > 75:
            strengths.append("Contacto visual consistente y directo")
        elif analysis.get('gaze', {}).get('gaze_centered_percentage', 0) < 50:
            improvements.append("Mantener mayor contacto visual con la cámara")
        
        # Distance analysis
        if analysis.get('distance', {}).get('optimal_percentage', 0) > 75:
            strengths.append("Distancia apropiada de cámara mantenida")
        elif analysis.get('distance', {}).get('optimal_percentage', 0) < 50:
            improvements.append("Ajustar y mantener distancia óptima de cámara")
        
        # Emotional analysis
        if analysis.get('emotions', {}).get('avg_valence', 0) > 0.2:
            strengths.append("Estado emocional positivo y energía apropiada")
        elif analysis.get('emotions', {}).get('avg_valence', 0) < -0.1:
            improvements.append("Trabajar en proyección de energía más positiva")
        
        # Hand analysis
        face_covered = analysis.get('hands', {}).get('face_covered_percentage', 0)
        if face_covered > 15:
            improvements.append("Evitar gestos que cubran el rostro")
        elif analysis.get('hands', {}).get('hands_visible_percentage', 0) > 60:
            strengths.append("Uso apropiado de gestos y comunicación no verbal")
        
        # Build basic report
        overall_score = analysis.get('overall', {}).get('overall_score', 0.5)
        performance = analysis.get('overall', {}).get('performance_level', 'moderate')
        
        basic_analysis = f"""RESUMEN EJECUTIVO:
Sesión de {analysis['session_info']['duration_minutes']:.1f} minutos completada con desempeño {performance.upper()}. 
Puntuación general: {overall_score:.2f}/1.0. {'Buen trabajo general con áreas específicas de mejora.' if overall_score > 0.6 else 'Oportunidades claras de mejora identificadas.'}

FORTALEZAS PRINCIPALES:"""

        for i, strength in enumerate(strengths[:3], 1):
            basic_analysis += f"\n{i}. {strength}"
        
        if not strengths:
            basic_analysis += "\n1. Sesión completada con participación activa"
        
        basic_analysis += "\n\nÁREAS DE MEJORA PRIORITARIAS:"
        
        for i, improvement in enumerate(improvements[:3], 1):
            basic_analysis += f"\n{i}. {improvement}"
        
        if not improvements:
            basic_analysis += "\n1. Mantener consistencia en el desempeño actual"
        
        basic_analysis += "\n\nRECOMENDACIÓN CLAVE:\n"
        if improvements:
            basic_analysis += f"Enfócate en: {improvements[0]}"
        else:
            basic_analysis += "Continúa practicando para mantener tu buen nivel"
        
        return self._format_final_report(basic_analysis, analysis)
    
    def _prepare_analysis_context(self, analysis: Dict) -> str:
        """Prepare context summary for ChatGPT."""
        context = f"""MÉTRICAS DE SESIÓN DE COACHING VIRTUAL:

Duración: {analysis['session_info']['duration_minutes']:.1f} minutos
Muestras analizadas: {analysis['session_info']['total_samples']}

RENDIMIENTO POR ÁREA:
"""
        
        if 'emotions' in analysis:
            context += f"• Emociones: {analysis['emotions'].get('most_common_emotion', 'N/A')} dominante, valencia {analysis['emotions'].get('avg_valence', 0):.2f} ({'positiva' if analysis['emotions'].get('avg_valence', 0) > 0 else 'negativa'})\n"
        
        if 'posture' in analysis:
            context += f"• Postura: {analysis['posture'].get('avg_posture_score', 0):.2f}/1.0, {analysis['posture'].get('frontal_percentage', 0):.1f}% frontal\n"
        
        if 'gaze' in analysis:
            context += f"• Contacto visual: {analysis['gaze'].get('gaze_centered_percentage', 0):.1f}% centrado\n"
        
        if 'distance' in analysis:
            context += f"• Distancia: {analysis['distance'].get('optimal_percentage', 0):.1f}% óptima, promedio {analysis['distance'].get('avg_distance_cm', 0):.0f}cm\n"
        
        if 'hands' in analysis:
            context += f"• Manos: {analysis['hands'].get('hands_visible_percentage', 0):.1f}% visibles, {analysis['hands'].get('face_covered_percentage', 0):.1f}% cubriendo rostro\n"
        
        context += f"\nPUNTUACIÓN GENERAL: {analysis.get('overall', {}).get('overall_score', 0):.2f}/1.0 ({analysis.get('overall', {}).get('performance_level', 'moderate')})"
        
        return context
    
    def _format_final_report(self, analysis_text: str, metrics: Dict) -> str:
        """Format the final report with analysis and detailed metrics."""
        
        report = f"""{'='*60}
REPORTE FINAL DE COACHING VIRTUAL
{'='*60}

{analysis_text}

{'='*60}
MÉTRICAS DETALLADAS DE LA SESIÓN
{'='*60}

INFORMACIÓN GENERAL:
• Duración: {metrics['session_info']['duration_minutes']:.1f} minutos
• Muestras analizadas: {metrics['session_info']['total_samples']} (cada {self.sample_interval}s)
• Puntuación general: {metrics.get('overall', {}).get('overall_score', 0):.3f}/1.0
"""
        
        # Emotional metrics
        if 'emotions' in metrics and metrics['emotions'].get('status') != 'no_data':
            emotions = metrics['emotions']
            report += f"""
ANÁLISIS EMOCIONAL:
• Emoción predominante: {emotions.get('most_common_emotion', 'N/A')} ({emotions.get('emotion_percentage', 0):.1f}% del tiempo)
• Valencia promedio: {emotions.get('avg_valence', 0):.3f} (rango: -1 negativa, +1 positiva)
• Nivel de arousal: {emotions.get('avg_arousal', 0):.3f} (energía emocional)
• Confianza promedio: {emotions.get('avg_confidence', 0):.3f}
• Calidad del humor: {emotions.get('mood_quality', 'N/A').upper()}
• Nivel de energía: {emotions.get('energy_level', 'N/A').upper()}
"""
        
        # Posture metrics
        if 'posture' in metrics and metrics['posture'].get('status') != 'no_data':
            posture = metrics['posture']
            report += f"""
ANÁLISIS DE POSTURA:
• Puntuación promedio: {posture.get('avg_posture_score', 0):.3f}/1.0
• Calidad general: {posture.get('posture_quality', 'N/A').upper()}
• Tiempo en orientación frontal: {posture.get('frontal_percentage', 0):.1f}%
• Simetría de hombros: {posture.get('shoulder_symmetry_percentage', 0):.1f}% del tiempo
• Consistencia postural: {posture.get('posture_consistency', 0):.3f}
• Mejor puntuación: {posture.get('best_score', 0):.3f}
"""
        
        # Gaze metrics
        if 'gaze' in metrics and metrics['gaze'].get('status') != 'no_data':
            gaze = metrics['gaze']
            report += f"""
ANÁLISIS DE CONTACTO VISUAL:
• Mirada centrada: {gaze.get('gaze_centered_percentage', 0):.1f}% del tiempo
• Calidad del contacto visual: {gaze.get('gaze_quality', 'N/A').upper()}
• Puntuación de centrado: {gaze.get('avg_centering_score', 0):.3f}
• Apertura ocular promedio: {gaze.get('avg_eye_aperture', 0):.3f}
• Nivel de alerta: {gaze.get('alertness_level', 'N/A').upper()}
• Total de parpadeos: {gaze.get('total_blinks', 0)}
"""
        
        # Distance metrics
        if 'distance' in metrics and metrics['distance'].get('status') != 'no_data':
            distance = metrics['distance']
            report += f"""
ANÁLISIS DE DISTANCIA:
• Distancia óptima: {distance.get('optimal_percentage', 0):.1f}% del tiempo
• Distancia promedio: {distance.get('avg_distance_cm', 0):.1f} cm
• Calidad de distancia: {distance.get('distance_quality', 'N/A').upper()}
• Estado más común: {distance.get('most_common_status', 'N/A').upper()}
"""
        
        # Hand metrics
        if 'hands' in metrics and metrics['hands'].get('status') != 'no_data':
            hands = metrics['hands']
            report += f"""
ANÁLISIS DE COMUNICACIÓN NO VERBAL:
• Manos visibles: {hands.get('hands_visible_percentage', 0):.1f}% del tiempo
• Rostro cubierto: {hands.get('face_covered_percentage', 0):.1f}% del tiempo
• Calidad de uso de manos: {hands.get('hand_usage_quality', 'N/A').upper()}
• Actividad gestual promedio: {hands.get('avg_gesture_activity', 0):.3f}
• Nivel de gesticulación: {hands.get('gesture_engagement', 'N/A').upper()}
• Gesto más común: {hands.get('most_common_gesture', 'N/A').upper()}
"""
        
        # Component scores
        if 'overall' in metrics and 'component_scores' in metrics['overall']:
            report += f"""
PUNTUACIONES POR COMPONENTE:"""
            for component, score in metrics['overall']['component_scores'].items():
                report += f"""
• {component.capitalize()}: {score:.3f}/1.0"""
        
        report += f"""

{'='*60}
Reporte generado automáticamente el {time.strftime('%Y-%m-%d %H:%M:%S')}
Sistema de Coaching Virtual con IA
{'='*60}
"""
        
        return report
    
    def save_session_data(self, filename: Optional[str] = None) -> str:
        """
        Save raw session data to JSON for further analysis.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Filename of saved data
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"session_data_{timestamp}.json"
        
        session_data = {
            'session_info': {
                'start_time': self.session_start_time,
                'duration_seconds': time.time() - self.session_start_time if self.session_start_time else 0,
                'sample_interval': self.sample_interval,
                'total_samples': len(self.session_metrics)
            },
            'metrics': self.session_metrics,
            'analysis': self.analyze_session() if self.session_metrics else None
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, default=str)
            print(f"📋 Session data saved: {filename}")
            return filename
        except Exception as e:
            print(f"Failed to save session data: {e}")
            return ""
    
    def get_quick_summary(self) -> str:
        """Get a quick one-line summary of the session."""
        if not self.session_metrics:
            return "No session data available"
        
        analysis = self.analyze_session()
        duration = analysis['session_info']['duration_minutes']
        performance = analysis.get('overall', {}).get('performance_level', 'moderate')
        score = analysis.get('overall', {}).get('overall_score', 0.5)
        
        return f"Session: {duration:.1f}min, Performance: {performance.upper()} ({score:.2f}/1.0)"