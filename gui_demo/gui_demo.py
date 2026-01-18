"""
UniNDP GUI Demo - Single Operator Compilation Visualization
============================================================
A Gradio-based GUI for demonstrating the UniNDP compiler's single operator compilation process.
"""

import gradio as gr
import os
import sys
import csv
import yaml
import subprocess
import threading
import queue
import time
import math
from io import StringIO

# Add parent directory (UniNDP root) to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

# ============================================================================
# Configuration and Constants
# ============================================================================

# Paths relative to UniNDP root (parent directory)
WORKLOAD_DIR = os.path.join(PARENT_DIR, 'workload')
CONFIG_DIR = os.path.join(PARENT_DIR, 'config')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

ARCHITECTURE_MAP = {
    'AIM (GDDR6)': 'aim',
    'AIM-8 (GDDR6)': 'aim8',
    'HBM-PIM': 'hbm-pim',
    'UPMEM': 'upmem',
    'DIMMining': 'dimmining'
}

# Only support mm type in this demo
WORKLOAD_TYPE = 'mm'

CUSTOM_CSS = """
    .gradio-container { 
        max-width: 100% !important; 
        padding: 10px 20px !important;
    }
    .main-content {
        max-width: 1800px;
        margin: 0 auto;
    }
    .panel-header { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-weight: bold;
        font-size: 1rem;
    }
    .contain { min-height: 0 !important; }
"""

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_typical_workloads():
    """Load typical workload sizes from CSV files (only mm type, excluding testsim.yaml)"""
    workloads = {}
    
    if not os.path.exists(WORKLOAD_DIR):
        return workloads
    
    for filename in os.listdir(WORKLOAD_DIR):
        if filename.endswith('.csv') and filename != 'testsim.yaml':
            filepath = os.path.join(WORKLOAD_DIR, filename)
            # Get file prefix for naming (e.g., "llama2_7B_prefill" from "llama2_7B_prefill.csv")
            file_prefix = filename.replace('.csv', '').replace('_n', '')
            try:
                with open(filepath, 'r') as f:
                    reader = csv.reader(f)
                    row_idx = 0
                    for row in reader:
                        if len(row) >= 6 and row[0].strip():
                            name = row[0].strip()
                            op_type = row[1].strip()
                            # Only load mm type workloads
                            if op_type != 'mm':
                                row_idx += 1
                                continue
                            try:
                                sizes = [int(row[i]) for i in range(2, 6)]
                                # Add file prefix to distinguish workloads, no batch display
                                key = f"[{file_prefix}] {name} (M={sizes[0]}, K={sizes[1]}, N={sizes[2]})"
                                workloads[key] = {
                                    'name': name,
                                    'type': op_type,
                                    'sizes': sizes[:3],  # Only M, K, N
                                    'file_order': row_idx  # Preserve original file order
                                }
                                row_idx += 1
                            except (ValueError, IndexError):
                                row_idx += 1
                                continue
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return workloads

def load_config_files():
    """Load available configuration files (excluding testsim.yaml)"""
    configs = {}
    
    if not os.path.exists(CONFIG_DIR):
        return configs
    
    for filename in os.listdir(CONFIG_DIR):
        if filename.endswith('.yaml') and filename != 'testsim.yaml':
            filepath = os.path.join(CONFIG_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    config_data = yaml.safe_load(f)
                    configs[filename] = config_data
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return configs

def get_config_for_architecture(arch_name):
    """Get the config file for a given architecture"""
    arch_key = ARCHITECTURE_MAP.get(arch_name, 'aim')
    config_map = {
        'aim': 'gddr6-aim.yaml',
        'aim8': 'gddr6-aim.yaml',
        'hbm-pim': 'hbm-pim.yaml',
        'upmem': 'upmem.yaml',
        'dimmining': 'dimmining.yaml'
    }
    return config_map.get(arch_key, 'gddr6-aim.yaml')

# ============================================================================
# Visualization Functions  
# ============================================================================

def generate_mm_svg(m, k, n, title="Matrix Multiplication (mm)"):
    """Generate SVG visualization of matrix multiplication (no batch display)"""
    # Scale factors - use viewBox for responsiveness
    scale = min(150 / max(m, k, n, 1), 50)
    m_h = max(20, min(m * scale / 50, 100))
    k_w = max(30, min(k * scale / 50, 120))
    n_w = max(30, min(n * scale / 50, 120))
    
    # Calculate total width needed
    total_width = 30 + k_w + 70 + n_w + 70 + n_w + 40
    
    svg = f'''<svg viewBox="0 0 {total_width} 150" xmlns="http://www.w3.org/2000/svg" style="width: 100%; max-width: 500px; height: auto;">
        <defs>
            <linearGradient id="inputGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
            </linearGradient>
            <linearGradient id="weightGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#f093fb;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#f5576c;stop-opacity:1" />
            </linearGradient>
            <linearGradient id="outputGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#4facfe;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#00f2fe;stop-opacity:1" />
            </linearGradient>
        </defs>
        
        <!-- Title -->
        <text x="{total_width/2}" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">{title}</text>
        
        <!-- Input Matrix A (M x K) -->
        <rect x="20" y="40" width="{k_w}" height="{m_h}" rx="4" fill="url(#inputGrad)" stroke="#5a67d8" stroke-width="2"/>
        <text x="{20 + k_w/2}" y="{40 + m_h/2 + 4}" text-anchor="middle" fill="white" font-size="11" font-weight="bold">A</text>
        <text x="{20 + k_w/2}" y="{50 + m_h + 12}" text-anchor="middle" fill="#666" font-size="10">{m}×{k}</text>
        <text x="{20 + k_w/2}" y="{50 + m_h + 24}" text-anchor="middle" fill="#888" font-size="9">Input</text>
        
        <!-- Multiply Symbol -->
        <text x="{35 + k_w + 15}" y="{40 + m_h/2 + 5}" text-anchor="middle" fill="#333" font-size="20">×</text>
        
        <!-- Weight Matrix B (K x N) -->
        <rect x="{60 + k_w + 20}" y="40" width="{n_w}" height="{max(20, min(k * scale / 50, 100))}" rx="4" fill="url(#weightGrad)" stroke="#ed64a6" stroke-width="2"/>
        <text x="{60 + k_w + 20 + n_w/2}" y="{40 + max(20, min(k * scale / 50, 100))/2 + 4}" text-anchor="middle" fill="white" font-size="11" font-weight="bold">B</text>
        <text x="{60 + k_w + 20 + n_w/2}" y="{50 + max(20, min(k * scale / 50, 100)) + 12}" text-anchor="middle" fill="#666" font-size="10">{k}×{n}</text>
        <text x="{60 + k_w + 20 + n_w/2}" y="{50 + max(20, min(k * scale / 50, 100)) + 24}" text-anchor="middle" fill="#888" font-size="9">Weight</text>
        
        <!-- Equals Symbol -->
        <text x="{95 + k_w + n_w + 25}" y="{40 + m_h/2 + 5}" text-anchor="middle" fill="#333" font-size="20">=</text>
        
        <!-- Output Matrix C (M x N) -->
        <rect x="{130 + k_w + n_w + 20}" y="40" width="{n_w}" height="{m_h}" rx="4" fill="url(#outputGrad)" stroke="#38b2ac" stroke-width="2"/>
        <text x="{130 + k_w + n_w + 20 + n_w/2}" y="{40 + m_h/2 + 4}" text-anchor="middle" fill="white" font-size="11" font-weight="bold">C</text>
        <text x="{130 + k_w + n_w + 20 + n_w/2}" y="{50 + m_h + 12}" text-anchor="middle" fill="#666" font-size="10">{m}×{n}</text>
        <text x="{130 + k_w + n_w + 20 + n_w/2}" y="{50 + m_h + 24}" text-anchor="middle" fill="#888" font-size="9">Output</text>
    </svg>'''
    
    return f'<div style="display: flex; justify-content: center;">{svg}</div>'

def generate_hardware_html(config, arch_name):
    """Generate HTML visualization of hardware architecture - compact hierarchical view"""
    ch = config.get('ch', 1)
    ra = config.get('ra', 1)
    de = config.get('de', 1)
    bg = config.get('bg', 4)
    ba = config.get('ba', 4)
    banks = bg * ba
    
    # Determine PU level and count based on architecture
    is_dimmining = 'dimmining' in arch_name.lower()
    is_aim8 = 'aim-8' in arch_name.lower() or 'aim8' in arch_name.lower()
    
    de_pu = config.get('de_pu', [0])
    if isinstance(de_pu, list):
        # For AIM: de_pu = [16, 8], index 0 for AIM, index 1 for AIM-8
        if is_aim8 and len(de_pu) > 1:
            de_pu_num = de_pu[1]  # AIM-8 uses 8 PUs
        else:
            de_pu_num = de_pu[0] if de_pu else 0
    else:
        de_pu_num = de_pu
    ra_pu = config.get('ra_pu', 0)
    
    pu_level = "Rank" if is_dimmining or ra_pu > 0 else "Device"
    pu_num = ra_pu if is_dimmining else de_pu_num
    total_pus = ch * ra * (1 if is_dimmining else de) * pu_num if pu_num > 0 else 0
    
    # Generate hierarchical HTML visualization
    html = f'''
    <div style="padding: 20px; background: linear-gradient(135deg, #f8fafc, #e2e8f0); border-radius: 12px; font-family: 'Segoe UI', sans-serif;">
        <!-- Header -->
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 1.3rem; font-weight: bold; color: #1a365d;">{arch_name}</div>
            <div style="font-size: 0.95rem; color: #4a5568; margin-top: 8px;">
                PU Level: <span style="color: #667eea; font-weight: 600;">{pu_level}</span> | 
                Total PUs: <span style="color: #667eea; font-weight: 600;">{total_pus}</span>
            </div>
        </div>
        
        <!-- Hierarchy Structure -->
        <div style="display: flex; flex-direction: column; gap: 10px;">
    '''
    
    # System Level
    html += f'''
            <!-- System Level -->
            <div style="background: linear-gradient(135deg, #4299e1, #3182ce); color: white; padding: 12px 18px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.4rem;">🖥️</span>
                    <span style="font-weight: 600; font-size: 1.05rem;">System</span>
                </div>
                <div style="display: flex; gap: 15px; font-size: 0.95rem;">
                    <span>Channels: <b>{ch}</b></span>
                </div>
            </div>
    '''
    
    # Channel Level
    html += f'''
            <!-- Channel Level -->
            <div style="margin-left: 25px; background: linear-gradient(135deg, #48bb78, #38a169); color: white; padding: 12px 18px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.3rem;">📡</span>
                    <span style="font-weight: 600; font-size: 1rem;">Channel</span>
                    <span style="opacity: 0.8; font-size: 0.9rem;">×{ch}</span>
                </div>
                <div style="display: flex; gap: 15px; font-size: 0.95rem;">
                    <span>Ranks/CH: <b>{ra}</b></span>
                </div>
            </div>
        '''
        
    # Rank Level
    html += f'''
            <!-- Rank Level -->
            <div style="margin-left: 50px; background: linear-gradient(135deg, #ed8936, #dd6b20); color: white; padding: 12px 18px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.3rem;">📦</span>
                    <span style="font-weight: 600; font-size: 1rem;">Rank</span>
                    <span style="opacity: 0.8; font-size: 0.9rem;">×{ra}</span>
                </div>
                <div style="display: flex; gap: 15px; font-size: 0.95rem;">
                    <span>Devices/Rank: <b>{de}</b></span>
                    {f'<span>PUs/Rank: <b>{ra_pu}</b></span>' if is_dimmining else ''}
                </div>
            </div>
    '''
    
    if not is_dimmining:
        # Device Level
        html += f'''
            <!-- Device Level -->
            <div style="margin-left: 75px; background: linear-gradient(135deg, #9f7aea, #805ad5); color: white; padding: 12px 18px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.3rem;">💾</span>
                    <span style="font-weight: 600; font-size: 1rem;">Device</span>
                    <span style="opacity: 0.8; font-size: 0.9rem;">×{de}</span>
                </div>
                <div style="display: flex; gap: 15px; font-size: 0.95rem;">
                    <span>PUs/Device: <b>{de_pu_num}</b></span>
                </div>
            </div>
                    '''
        
        # PU Level
        if de_pu_num > 0:
            html += f'''
            <!-- PU Level -->
            <div style="margin-left: 100px; background: linear-gradient(135deg, #f56565, #e53e3e); color: white; padding: 12px 18px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.3rem;">⚡</span>
                    <span style="font-weight: 600; font-size: 1rem;">Processing Unit</span>
                    <span style="opacity: 1; font-size: 1rem;">×{de_pu_num}</span>
                </div>
            </div>
                            '''
    
        # Bank Level (same indent as PU)
        html += f'''
            <!-- Bank Level -->
            <div style="margin-left: 100px; background: linear-gradient(135deg, #718096, #4a5568); color: white; padding: 12px 18px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.3rem;">🗃️</span>
                    <span style="font-weight: 600; font-size: 1rem;">Banks</span>
                    <span style="opacity: 1; font-size: 1rem;">×{banks}</span>
                </div>
            </div>
        '''
    else:
        # DIMMining: PU at rank level
        if ra_pu > 0:
            html += f'''
            <!-- PU Level (at Rank) -->
            <div style="margin-left: 75px; background: linear-gradient(135deg, #f56565, #e53e3e); color: white; padding: 12px 18px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.3rem;">⚡</span>
                    <span style="font-weight: 600; font-size: 1rem;">Processing Unit</span>
                    <span style="opacity: 1; font-size: 1rem;">×{ra_pu}</span>
                </div>
            </div>
    '''
    
        # Device Level (same indent as PU for DIMMining)
        html += f'''
            <!-- Device Level -->
            <div style="margin-left: 75px; background: linear-gradient(135deg, #9f7aea, #805ad5); color: white; padding: 12px 18px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.3rem;">💾</span>
                    <span style="font-weight: 600; font-size: 1rem;">Device</span>
                    <span style="opacity: 0.8; font-size: 0.9rem;">×{de}</span>
                </div>
            </div>
        '''
        
        # Bank Level (more indent for DIMMining)
        html += f'''
            <!-- Bank Level -->
            <div style="margin-left: 100px; background: linear-gradient(135deg, #718096, #4a5568); color: white; padding: 12px 18px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.3rem;">🗃️</span>
                    <span style="font-weight: 600; font-size: 1rem;">Banks</span>
                    <span style="opacity: 0.8; font-size: 0.9rem;">×{banks}</span>
                </div>
            </div>
        '''
        
    # Close the hierarchy div and main container
    html += '''
        </div>
    </div>
    '''
    
    return html

def generate_performance_html(result):
    """Generate HTML visualization for performance comparison with detailed metrics"""
    
    # Extract basic metrics
    best_cycles = result.get('best_cycles', result.get('optimal_lat', 0))
    baseline_cycles = result.get('baseline_cycles', result.get('baseline_lat', 0))
    speedup = result.get('speedup', 1.0)
    
    # Extract latency breakdown
    baseline_compute = result.get('baseline_compute_lat', 0)
    baseline_row_change = result.get('baseline_row_change_lat', 0)
    baseline_host_access = result.get('baseline_host_access_lat', 0)
    
    best_compute = result.get('best_compute_lat', 0)
    best_row_change = result.get('best_row_change_lat', 0)
    best_host_access = result.get('best_host_access_lat', 0)
    
    # Check if we have breakdown data
    has_breakdown = (baseline_compute > 0 or baseline_row_change > 0 or baseline_host_access > 0 or
                     best_compute > 0 or best_row_change > 0 or best_host_access > 0)
    
    # Extract detailed metrics from CSV
    metrics = {
        'lat': {
            'name': 'Latency',
            'unit': 'cycles',
            'optimal': result.get('optimal_lat', best_cycles),
            'baseline': result.get('baseline_lat', baseline_cycles),
            'icon': '⏱️',
            'lower_better': True
        },
        'cmd': {
            'name': 'Commands',
            'unit': '',
            'optimal': result.get('optimal_cmd', 0),
            'baseline': result.get('baseline_cmd', 0),
            'icon': '📤',
            'lower_better': True
        },
        'pu_dram': {
            'name': 'PU DRAM Access',
            'unit': '',
            'optimal': result.get('optimal_pu_dram', 0),
            'baseline': result.get('baseline_pu_dram', 0),
            'icon': '💾',
            'lower_better': True
        },
        'host_dram': {
            'name': 'Host DRAM Access',
            'unit': '',
            'optimal': result.get('optimal_host_dram', 0),
            'baseline': result.get('baseline_host_dram', 0),
            'icon': '🖥️',
            'lower_better': True
        },
        'row_change': {
            'name': 'Row Changes',
            'unit': '',
            'optimal': result.get('optimal_row_change', 0),
            'baseline': result.get('baseline_row_change', 0),
            'icon': '🔄',
            'lower_better': True
        }
    }
    
    max_cycles = max(best_cycles, baseline_cycles, 1)
    best_pct = (best_cycles / max_cycles) * 100
    baseline_pct = (baseline_cycles / max_cycles) * 100
    
    # Speedup color based on value
    if speedup >= 2.0:
        speedup_bg = 'linear-gradient(135deg, #38a169, #68d391)'
    elif speedup >= 1.5:
        speedup_bg = 'linear-gradient(135deg, #3182ce, #63b3ed)'
    elif speedup >= 1.0:
        speedup_bg = 'linear-gradient(135deg, #dd6b20, #f6ad55)'
    else:
        speedup_bg = 'linear-gradient(135deg, #e53e3e, #fc8181)'
    
    # Generate metric comparison rows
    def generate_metric_row(key, m):
        optimal = m['optimal']
        baseline = m['baseline']
        
        if baseline > 0:
            ratio = optimal / baseline
            if m['lower_better']:
                improvement = (1 - ratio) * 100
                if improvement > 0:
                    change_text = f'<span style="color: #38a169; font-weight: 600;">↓ {improvement:.1f}%</span>'
                elif improvement < 0:
                    change_text = f'<span style="color: #e53e3e; font-weight: 600;">↑ {-improvement:.1f}%</span>'
                else:
                    change_text = '<span style="color: #718096;">—</span>'
            else:
                improvement = (ratio - 1) * 100
                if improvement > 0:
                    change_text = f'<span style="color: #38a169; font-weight: 600;">↑ {improvement:.1f}%</span>'
                elif improvement < 0:
                    change_text = f'<span style="color: #e53e3e; font-weight: 600;">↓ {-improvement:.1f}%</span>'
                else:
                    change_text = '<span style="color: #718096;">—</span>'
        else:
            change_text = '<span style="color: #718096;">—</span>'
        
        unit_str = f' {m["unit"]}' if m['unit'] else ''
        
        return f'''
        <tr style="border-bottom: 1px solid #e2e8f0;">
            <td style="padding: 10px 12px; font-weight: 500; color: #4a5568;">
                <span style="margin-right: 8px;">{m['icon']}</span>{m['name']}
            </td>
            <td style="padding: 10px 12px; text-align: right; color: #718096; font-family: 'JetBrains Mono', monospace;">
                {baseline:,}{unit_str}
            </td>
            <td style="padding: 10px 12px; text-align: right; color: #38a169; font-family: 'JetBrains Mono', monospace; font-weight: 600;">
                {optimal:,}{unit_str}
            </td>
            <td style="padding: 10px 12px; text-align: right;">
                {change_text}
            </td>
        </tr>
        '''
    
    metric_rows = ''.join([generate_metric_row(k, v) for k, v in metrics.items() if v['optimal'] > 0 or v['baseline'] > 0])
    
    # Generate stacked bar chart for latency breakdown
    if has_breakdown:
        # Calculate percentages for stacked bars
        baseline_total = baseline_compute + baseline_row_change + baseline_host_access
        best_total = best_compute + best_row_change + best_host_access
        
        # Use total from breakdown if available, otherwise use cycles
        if baseline_total == 0:
            baseline_total = baseline_cycles
        if best_total == 0:
            best_total = best_cycles
            
        max_total = max(baseline_total, best_total, 1)
        
        # Calculate segment widths (relative to max_total)
        baseline_compute_w = (baseline_compute / max_total) * 100 if baseline_total > 0 else 0
        baseline_row_w = (baseline_row_change / max_total) * 100 if baseline_total > 0 else 0
        baseline_host_w = (baseline_host_access / max_total) * 100 if baseline_total > 0 else 0
        
        best_compute_w = (best_compute / max_total) * 100 if best_total > 0 else 0
        best_row_w = (best_row_change / max_total) * 100 if best_total > 0 else 0
        best_host_w = (best_host_access / max_total) * 100 if best_total > 0 else 0
        
        latency_bar_html = f'''
        <div style="background: white; border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="font-size: 0.9rem; font-weight: 600; color: #2d3748; margin-bottom: 12px;">⏱️ Latency Breakdown Comparison</div>
            
            <!-- Legend with values -->
            <div style="display: flex; gap: 20px; margin-bottom: 12px; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 14px; height: 14px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 3px;"></div>
                    <span style="font-size: 0.8rem; color: #4a5568;">Compute</span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 14px; height: 14px; background: linear-gradient(135deg, #f093fb, #f5576c); border-radius: 3px;"></div>
                    <span style="font-size: 0.8rem; color: #4a5568;">Row Change</span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <div style="width: 14px; height: 14px; background: linear-gradient(135deg, #4facfe, #00f2fe); border-radius: 3px;"></div>
                    <span style="font-size: 0.8rem; color: #4a5568;">Input and Output</span>
                </div>
            </div>
        
            <!-- Baseline Stacked Bar -->
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-size: 0.85rem; color: #718096;">Baseline</span>
                    <span style="font-size: 0.85rem; color: #718096; font-family: 'JetBrains Mono', monospace;">{baseline_total:,.0f} cycles</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 6px; height: 32px; overflow: visible; display: flex; position: relative;">
                    <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; width: {baseline_compute_w}%; position: relative; border-radius: 6px 0 0 6px;" title="Compute: {baseline_compute:,.0f}">
                        <span style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); color: white; font-size: 0.7rem; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.5); white-space: nowrap; z-index: 10;">{baseline_compute:,.0f}</span>
                    </div>
                    <div style="background: linear-gradient(90deg, #f093fb, #f5576c); height: 100%; width: {baseline_row_w}%; position: relative;" title="Row Change: {baseline_row_change:,.0f}">
                        <span style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); color: white; font-size: 0.7rem; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.5); white-space: nowrap; z-index: 10;">{baseline_row_change:,.0f}</span>
                    </div>
                    <div style="background: linear-gradient(90deg, #4facfe, #00f2fe); height: 100%; width: {baseline_host_w}%; position: relative; border-radius: 0 6px 6px 0;" title="Host Access: {baseline_host_access:,.0f}">
                        <span style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); color: white; font-size: 0.7rem; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.5); white-space: nowrap; z-index: 10;">{baseline_host_access:,.0f}</span>
                    </div>
                </div>
            </div>
        
            <!-- Optimized Stacked Bar -->
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-size: 0.85rem; color: #38a169; font-weight: 600;">Optimized</span>
                    <span style="font-size: 0.85rem; color: #38a169; font-family: 'JetBrains Mono', monospace; font-weight: 600;">{best_total:,.0f} cycles</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 6px; height: 32px; overflow: visible; display: flex; position: relative;">
                    <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; width: {best_compute_w}%; position: relative; border-radius: 6px 0 0 6px;" title="Compute: {best_compute:,.0f}">
                        <span style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); color: white; font-size: 0.7rem; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.5); white-space: nowrap; z-index: 10;">{best_compute:,.0f}</span>
                    </div>
                    <div style="background: linear-gradient(90deg, #f093fb, #f5576c); height: 100%; width: {best_row_w}%; position: relative;" title="Row Change: {best_row_change:,.0f}">
                        <span style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); color: white; font-size: 0.7rem; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.5); white-space: nowrap; z-index: 10;">{best_row_change:,.0f}</span>
                    </div>
                    <div style="background: linear-gradient(90deg, #4facfe, #00f2fe); height: 100%; width: {best_host_w}%; position: relative; border-radius: 0 6px 6px 0;" title="Host Access: {best_host_access:,.0f}">
                        <span style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); color: white; font-size: 0.7rem; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.5); white-space: nowrap; z-index: 10;">{best_host_access:,.0f}</span>
                    </div>
                </div>
            </div>
            
            <!-- Breakdown Details Table -->
            <div style="margin-top: 15px; border-top: 1px solid #e2e8f0; padding-top: 12px;">
                <table style="width: 100%; font-size: 0.8rem; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 1px solid #e2e8f0;">
                            <th style="padding: 6px; text-align: left; color: #4a5568;">Component</th>
                            <th style="padding: 6px; text-align: right; color: #718096;">Baseline</th>
                            <th style="padding: 6px; text-align: right; color: #38a169;">Optimized</th>
                            <th style="padding: 6px; text-align: right; color: #4a5568;">Change</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid #f0f0f0;">
                            <td style="padding: 6px; color: #667eea;">⚡ Compute</td>
                            <td style="padding: 6px; text-align: right; font-family: monospace;">{baseline_compute:,.0f}</td>
                            <td style="padding: 6px; text-align: right; font-family: monospace;">{best_compute:,.0f}</td>
                            <td style="padding: 6px; text-align: right;">{f'<span style="color: #38a169;">↓{((baseline_compute - best_compute) / baseline_compute * 100):.1f}%</span>' if baseline_compute > 0 and best_compute < baseline_compute else f'<span style="color: #e53e3e;">↑{((best_compute - baseline_compute) / baseline_compute * 100):.1f}%</span>' if baseline_compute > 0 and best_compute > baseline_compute else '—'}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #f0f0f0;">
                            <td style="padding: 6px; color: #f5576c;">🔄 Row Change</td>
                            <td style="padding: 6px; text-align: right; font-family: monospace;">{baseline_row_change:,.0f}</td>
                            <td style="padding: 6px; text-align: right; font-family: monospace;">{best_row_change:,.0f}</td>
                            <td style="padding: 6px; text-align: right;">{f'<span style="color: #38a169;">↓{((baseline_row_change - best_row_change) / baseline_row_change * 100):.1f}%</span>' if baseline_row_change > 0 and best_row_change < baseline_row_change else f'<span style="color: #e53e3e;">↑{((best_row_change - baseline_row_change) / baseline_row_change * 100):.1f}%</span>' if baseline_row_change > 0 and best_row_change > baseline_row_change else '—'}</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px; color: #00b4d8;">🖥️ Host Access</td>
                            <td style="padding: 6px; text-align: right; font-family: monospace;">{baseline_host_access:,.0f}</td>
                            <td style="padding: 6px; text-align: right; font-family: monospace;">{best_host_access:,.0f}</td>
                            <td style="padding: 6px; text-align: right;">{f'<span style="color: #38a169;">↓{((baseline_host_access - best_host_access) / baseline_host_access * 100):.1f}%</span>' if baseline_host_access > 0 and best_host_access < baseline_host_access else f'<span style="color: #e53e3e;">↑{((best_host_access - baseline_host_access) / baseline_host_access * 100):.1f}%</span>' if baseline_host_access > 0 and best_host_access > baseline_host_access else '—'}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        '''
    else:
        # Fallback to simple bar if no breakdown data
        latency_bar_html = f'''
        <div style="background: white; border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="font-size: 0.9rem; font-weight: 600; color: #2d3748; margin-bottom: 12px;">⏱️ Latency Comparison</div>
        
            <!-- Baseline Bar -->
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-size: 0.85rem; color: #718096;">Baseline</span>
                    <span style="font-size: 0.85rem; color: #718096; font-family: 'JetBrains Mono', monospace;">{baseline_cycles:,.0f} cycles</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 6px; height: 24px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #718096, #a0aec0); height: 100%; width: {baseline_pct}%; border-radius: 6px; transition: width 0.5s ease;"></div>
                </div>
            </div>
        
            <!-- Optimized Bar -->
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-size: 0.85rem; color: #38a169; font-weight: 600;">Optimized</span>
                    <span style="font-size: 0.85rem; color: #38a169; font-family: 'JetBrains Mono', monospace; font-weight: 600;">{best_cycles:,.0f} cycles</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 6px; height: 24px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #38a169, #68d391); height: 100%; width: {best_pct}%; border-radius: 6px; transition: width 0.5s ease;"></div>
                </div>
            </div>
        </div>
        '''
    
    html = f'''
    <div style="padding: 20px; background: linear-gradient(135deg, #f8fafc, #e2e8f0); border-radius: 12px;">
        
        <!-- Speedup Hero Display -->
        <div style="background: {speedup_bg}; border-radius: 15px; padding: 25px; text-align: center; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin-bottom: 20px;">
            <div style="font-size: 1rem; opacity: 0.9; margin-bottom: 5px;">🚀 Speedup</div>
            <div style="font-size: 3.5rem; font-weight: bold; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{speedup:.2f}×</div>
        </div>
        
        {latency_bar_html}
        
        <!-- Detailed Metrics Table -->
        <div style="background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="padding: 12px 15px; background: linear-gradient(90deg, #2d3748, #4a5568); color: white; font-size: 0.9rem; font-weight: 600;">
                📊 Detailed Metrics Comparison
            </div>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
                <thead>
                    <tr style="background: #f7fafc; border-bottom: 2px solid #e2e8f0;">
                        <th style="padding: 10px 12px; text-align: left; font-weight: 600; color: #4a5568;">Metric</th>
                        <th style="padding: 10px 12px; text-align: right; font-weight: 600; color: #718096;">Baseline</th>
                        <th style="padding: 10px 12px; text-align: right; font-weight: 600; color: #38a169;">Optimized</th>
                        <th style="padding: 10px 12px; text-align: right; font-weight: 600; color: #4a5568;">Change</th>
                    </tr>
                </thead>
                <tbody>
                    {metric_rows}
                </tbody>
            </table>
        </div>
    </div>
    '''
    
    return html

def parse_best_design(design_str, arch_name):
    """Parse the best design string to extract partition information"""
    try:
        import re
        
        design_str = design_str.strip()
        if design_str.startswith('[') and design_str.endswith(']'):
            design_str = design_str[1:-1]
        
        # Extract tuples pattern
        partition_match = re.search(r'\(\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)(?:,\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\))?\)', design_str)
        
        if partition_match:
            groups = partition_match.groups()
            ch_div = (int(groups[0]), int(groups[1]), int(groups[2]), int(groups[3]))
            ra_div = (int(groups[4]), int(groups[5]), int(groups[6]), int(groups[7]))
            de_div = (int(groups[8]), int(groups[9]), int(groups[10]), int(groups[11]))
            
            if groups[12] is not None:
                pu_div = (int(groups[12]), int(groups[13]), int(groups[14]), int(groups[15]))
                return {
                    'ch': ch_div,
                    'ra': ra_div,
                    'de': de_div,
                    'pu': pu_div
                }
            else:
                return {
                    'ch': ch_div,
                    'ra': ra_div,
                    'pu': de_div
                }
        
        return None
    except Exception as e:
        print(f"Error parsing design: {e}")
        return None

def parse_best_design(design_str, arch_name):
    """Parse the best design string to extract partition information"""
    try:
        import re
        
        design_str = design_str.strip()
        if design_str.startswith('[') and design_str.endswith(']'):
            design_str = design_str[1:-1]
        
        # Extract tuples pattern
        partition_match = re.search(r'\(\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)(?:,\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\))?\)', design_str)
        
        if partition_match:
            groups = partition_match.groups()
            ch_div = (int(groups[0]), int(groups[1]), int(groups[2]), int(groups[3]))
            ra_div = (int(groups[4]), int(groups[5]), int(groups[6]), int(groups[7]))
            de_div = (int(groups[8]), int(groups[9]), int(groups[10]), int(groups[11]))
            
            if groups[12] is not None:
                pu_div = (int(groups[12]), int(groups[13]), int(groups[14]), int(groups[15]))
                return {
                    'ch': ch_div,
                    'ra': ra_div,
                    'de': de_div,
                    'pu': pu_div
                }
            else:
                return {
                    'ch': ch_div,
                    'ra': ra_div,
                    'pu': de_div
                }
        
        return None
    except Exception as e:
        print(f"Error parsing design: {e}")
        return None

def generate_detailed_partition_html(design_str, mm_size, config, arch_name):
    """Generate detailed HTML visualization of the partition strategy"""
    m, k, n, b = mm_size
    is_dimmining = 'dimmining' in arch_name.lower()
    
    parsed = parse_best_design(design_str, arch_name)
    
    if parsed is None:
        return f'''
        <div style="padding: 20px; background: linear-gradient(135deg, #f7fafc, #edf2f7); border-radius: 12px;">
            <h4 style="color: #2d3748; margin-bottom: 15px; display: flex; align-items: center; font-size: 1.15rem;">
                <span style="margin-right: 10px;">📋</span>
                Best Strategy Configuration
            </h4>
            <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea;">
                <pre style="white-space: pre-wrap; word-wrap: break-word; color: #4a5568; font-size: 0.85rem; font-family: 'Monaco', 'Consolas', monospace; margin: 0;">{design_str}</pre>
            </div>
        </div>
        '''
    
    levels_html = ""
    colors = ['#4299e1', '#48bb78', '#ed8936', '#9f7aea']
    level_icons = ['📡', '📦', '💾', '⚡']
    
    curr_m, curr_k, curr_n, curr_b = m, k, n, b
    
    if is_dimmining:
        level_names = ['Channel', 'Rank', 'PU']
        divs = [parsed.get('ch', (1,1,1,1)), parsed.get('ra', (1,1,1,1)), parsed.get('pu', (1,1,1,1))]
    else:
        level_names = ['Channel', 'Rank', 'Device', 'PU']
        divs = [parsed.get('ch', (1,1,1,1)), parsed.get('ra', (1,1,1,1)), parsed.get('de', (1,1,1,1)), parsed.get('pu', (1,1,1,1))]
    
    for i, (level_name, div) in enumerate(zip(level_names, divs)):
        div_m, div_k, div_n, div_b = div
        total_div = div_m * div_k * div_n * div_b
        
        new_m = math.ceil(curr_m / div_m) if div_m > 0 else curr_m
        new_k = math.ceil(curr_k / div_k) if div_k > 0 else curr_k
        new_n = math.ceil(curr_n / div_n) if div_n > 0 else curr_n
        new_b = math.ceil(curr_b / div_b) if div_b > 0 else curr_b
        
        color = colors[i % len(colors)]
        icon = level_icons[i % len(level_icons)]
        
        levels_html += f'''
        <div style="margin-bottom: 12px; padding: 15px; background: white; border-radius: 8px; border-left: 5px solid {color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.1rem;">{icon}</span>
                    <span style="color: {color}; font-weight: 600; font-size: 1.05rem;">{level_name}</span>
                </div>
                <span style="background: {color}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.85rem;">÷{total_div}</span>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; font-size: 0.9rem;">
                <div style="background: #f7fafc; padding: 10px; border-radius: 6px; text-align: center;">
                    <div style="color: #718096; margin-bottom: 4px; font-size: 0.8rem;">Division</div>
                    <div style="font-weight: 600; color: #2d3748;">M÷{div_m} K÷{div_k} N÷{div_n}</div>
                </div>
                <div style="background: #f7fafc; padding: 10px; border-radius: 6px; text-align: center;">
                    <div style="color: #718096; margin-bottom: 4px; font-size: 0.8rem;">Before</div>
                    <div style="font-weight: 600; color: #4a5568;">{curr_m}×{curr_k}×{curr_n}</div>
                </div>
                <div style="background: linear-gradient(135deg, {color}15, {color}08); padding: 10px; border-radius: 6px; text-align: center;">
                    <div style="color: #718096; margin-bottom: 4px; font-size: 0.8rem;">After</div>
                    <div style="font-weight: 600; color: {color};">{new_m}×{new_k}×{new_n}</div>
                </div>
            </div>
        </div>
        '''
        
        curr_m, curr_k, curr_n, curr_b = new_m, new_k, new_n, new_b
    
    final_html = f'''
    <div style="margin-top: 15px; padding: 18px; background: linear-gradient(135deg, #38a16915, #68d39115); border-radius: 10px; text-align: center; border: 2px solid #38a169;">
        <div style="font-size: 1rem; color: #276749; margin-bottom: 6px;">Final Sub-operator Size (per PU)</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #22543d;">M={curr_m} × K={curr_k} × N={curr_n}</div>
    </div>
    '''
    
    return f'''
    <div style="padding: 20px; background: linear-gradient(135deg, #f7fafc, #edf2f7); border-radius: 12px;">
        <h4 style="color: #2d3748; margin-bottom: 15px; display: flex; align-items: center; font-size: 1.15rem;">
            <span style="margin-right: 10px;">🎯</span>
            Partition Strategy
        </h4>
        <div style="margin-bottom: 15px; padding: 12px; background: white; border-radius: 8px; border: 1px solid #e2e8f0;">
            <div style="font-size: 0.9rem; color: #718096;">Original Size</div>
            <div style="font-size: 1.15rem; font-weight: bold; color: #2d3748;">M={m} × K={k} × N={n}</div>
        </div>
        {levels_html}
        {final_html}
    </div>
    '''

def parse_dram_mapping_info(design_str):
    """Parse DRAM mapping info from best design string"""
    import re
    
    try:
        if design_str.strip().startswith('['):
            first_design_match = re.search(r'\[(\(<LEVEL\.\w+:.*?\)\))\]', design_str)
            if first_design_match:
                design_str = first_design_match.group(1)
            else:
                match = re.search(r'(\(<LEVEL\.\w+:.*?\)\))', design_str)
                if match:
                    design_str = match.group(1)
        
        simd_matches = list(re.finditer(r'\),\s*(\d+),\s*\(\(', design_str))
        simd_k = int(simd_matches[0].group(1)) if len(simd_matches) > 0 else 1
        simd_l = int(simd_matches[1].group(1)) if len(simd_matches) > 1 else 1
        
        input_match = re.search(r'\),\s*\d+,\s*\(\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', design_str)
        if input_match:
            g = input_match.groups()
            input_block = (int(g[0]), int(g[1]), int(g[2]), int(g[3]))
            input_rows = (int(g[4]), int(g[5]), int(g[6]), int(g[7]))
        else:
            input_block = (1, 1, 1, 1)
            input_rows = (1, 1, 1, 1)
        
        output_match = re.search(r'\)\),\s*\d+,\s*\(\((\d+),\s*(\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+),\s*(\d+)\)', design_str)
        if output_match:
            g = output_match.groups()
            output_block = (int(g[0]), int(g[1]), int(g[2]))
            output_rows = (int(g[3]), int(g[4]), int(g[5]))
        else:
            output_block = (1, 1, 1)
            output_rows = (1, 1, 1)
        
        return {
            'simd_k': simd_k,
            'simd_l': simd_l,
            'input_block': input_block,
            'input_rows': input_rows,
            'output_block': output_block,
            'output_rows': output_rows
        }
    except Exception as e:
        print(f"Error parsing DRAM mapping: {e}")
        return None

def generate_dram_mapping_html(design_str, mm_size, config, arch_name):
    """Generate HTML visualization of DRAM row/column mapping for matrices A, B, C"""
    m, k, n, b = mm_size
    
    dram_info = parse_dram_mapping_info(design_str)
    
    if dram_info is None:
        return f'''
        <div style="padding: 15px; background: linear-gradient(135deg, #f7fafc, #edf2f7); border-radius: 12px;">
            <h4 style="color: #2d3748; margin-bottom: 10px;">📊 DRAM Mapping</h4>
            <div style="background: white; padding: 12px; border-radius: 8px; border-left: 4px solid #667eea;">
                <pre style="white-space: pre-wrap; word-wrap: break-word; color: #4a5568; font-size: 0.8rem; margin: 0;">{design_str}</pre>
            </div>
        </div>
        '''
    
    simd_k = dram_info['simd_k']
    simd_l = dram_info['simd_l']
    input_block = dram_info['input_block']
    input_rows = dram_info['input_rows']
    output_block = dram_info['output_block']
    output_rows = dram_info['output_rows']
    
    a_col_size = f"1×{simd_k}"
    a_row_size = f"{input_block[0]}×{input_block[1]*simd_k}"
    a_total_rows = input_rows[0] * input_rows[1]
    
    b_col_size = f"{simd_k}×1"
    b_row_size = f"{input_block[1]*simd_k}×{input_block[2]}"
    b_total_rows = input_rows[1] * input_rows[2]
    
    c_col_size = f"1x{simd_l}"
    c_row_size = f"{output_block[0]}×{output_block[1]*simd_l}"
    c_total_rows = output_rows[0] * output_rows[1]
    
    def matrix_card(name, color, dims, col_size, row_size, total_rows, icon):
        return f'''
        <div style="flex: 1; min-width: 180px; padding: 15px; background: white; border-radius: 10px; border-top: 4px solid {color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <span style="font-size: 1.3rem;">{icon}</span>
                <div>
                    <div style="font-weight: 700; color: {color}; font-size: 1.1rem;">Matrix {name}</div>
                    <div style="font-size: 0.8rem; color: #718096;">{dims}</div>
                </div>
            </div>
            
            <div style="space-y: 8px;">
                <div style="background: #f7fafc; padding: 8px 10px; border-radius: 6px; margin-bottom: 8px;">
                    <div style="font-size: 0.75rem; color: #718096;">Column Block Size</div>
                    <div style="font-weight: 600; color: #2d3748;">{col_size}</div>
                </div>
                <div style="background: #f7fafc; padding: 8px 10px; border-radius: 6px; margin-bottom: 8px;">
                    <div style="font-size: 0.75rem; color: #718096;">Row Block Size</div>
                    <div style="font-weight: 600; color: #2d3748;">{row_size}</div>
                </div>
                <div style="background: linear-gradient(135deg, {color}15, {color}08); padding: 8px 10px; border-radius: 6px;">
                    <div style="font-size: 0.75rem; color: #718096;">Total Rows Used</div>
                    <div style="font-weight: 700; color: {color}; font-size: 1.1rem;">{total_rows}</div>
                </div>
            </div>
        </div>
        '''
    
    html = f'''
    <div style="padding: 18px; background: linear-gradient(135deg, #f7fafc, #edf2f7); border-radius: 12px;">
        <h4 style="color: #2d3748; margin-bottom: 15px; display: flex; align-items: center; font-size: 1.1rem;">
            <span style="margin-right: 8px;">📊</span>
            DRAM Row/Column Mapping
        </h4>
        
        <div style="display: flex; gap: 12px; flex-wrap: wrap;">
            {matrix_card('A (Input)', '#667eea', 'M×K', a_col_size, a_row_size, a_total_rows, '📥')}
            {matrix_card('B (Weight)', '#f093fb', 'K×N', b_col_size, b_row_size, b_total_rows, '⚖️')}
            {matrix_card('C (Output)', '#4facfe', 'M×N', c_col_size, c_row_size, c_total_rows, '📤')}
        </div>
    </div>
    '''
    
    return html

# ============================================================================
# Compilation Runner with Streaming
# ============================================================================

class CompilationRunner:
    """Handles running the compilation process and capturing output"""
    
    def __init__(self):
        self.process = None
        self.is_running = False
        self.result = None
        self.should_stop = False
        
    def run_compilation_stream(self, arch, workload_type, sizes, options):
        """Run compilation as a generator for streaming output"""
        self.is_running = True
        self.should_stop = False
        self.result = None
        
        arch_map = {
            'AIM (GDDR6)': 'aim',
            'AIM-8 (GDDR6)': 'aim8',
            'HBM-PIM': 'hbm-pim',
            'UPMEM': 'upmem',
            'DIMMining': 'dimmining'
        }
        arch_key = arch_map.get(arch, 'upmem')
        wl_key = 'mm'
        
        cmd = [
            'python', '-u', os.path.join(PARENT_DIR, 'compile.py'),
            '-A', arch_key,
            '-W', wl_key,
            '-S', str(sizes[0]), str(sizes[1]), str(sizes[2]), str(sizes[3]),
            '-N', 'gui_demo',
            '-WS', OUTPUT_DIR,
            '-O', 'demo'
        ]
        
        # Add options
        if options.get('po2'):
            cmd.append('-P')
        if options.get('quicksearch'):
            cmd.append('-Q')
        if options.get('allow_under_utilize'):
            cmd.append('-UU')
        if options.get('topk'):
            cmd.extend(['-K', str(options['topk'])])
        if options.get('cmdthre'):
            cmd.extend(['-T', str(options['cmdthre'])])
        if options.get('datapr', 16) != 16:
            cmd.extend(['-D', str(options['datapr'])])
        if options.get('puinbuf', 0) > 0:
            cmd.extend(['-IB', str(options['puinbuf'])])
        if options.get('puoutbuf', 0) > 0:
            cmd.extend(['-OB', str(options['puoutbuf'])])
        if options.get('puslowdown', 1) > 1:
            cmd.extend(['-PS', str(options['puslowdown'])])
        
        output_lines = []
        
        cmd_str = ' '.join(cmd)
        output_lines.append(f"[DEBUG] Executing: {cmd_str}")
        output_lines.append("")
        
        try:
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=PARENT_DIR,
                env=env
            )
            
            while True:
                if self.should_stop:
                    self.process.terminate()
                    break
                    
                line = self.process.stdout.readline()
                if line == '' and self.process.poll() is not None:
                    break
                if line:
                    output_lines.append(line.rstrip())
                    yield '\n'.join(output_lines[-50:])
            
            self.process.wait()
            
        except Exception as e:
            output_lines.append(f"Error: {str(e)}")
            yield '\n'.join(output_lines)
        
        finally:
            self.is_running = False
            
            log_path = os.path.join(OUTPUT_DIR, 'demo', 'log', '_gui_demo.log')
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        log_content = f.read()
                        self.result = self._parse_log(log_content)
                except:
                    pass
    
    def _parse_log(self, log_content):
        """Parse the log file to extract results"""
        result = {}
        lines = log_content.split('\n')
        
        for line in lines:
            if 'baseline result:' in line:
                try:
                    result['baseline_cycles'] = float(line.split(':')[-1].strip())
                except:
                    pass
            elif 'baseline compute lat:' in line:
                try:
                    result['baseline_compute_lat'] = float(line.split(':')[-1].strip())
                except:
                    pass
            elif 'baseline row change lat:' in line:
                try:
                    result['baseline_row_change_lat'] = float(line.split(':')[-1].strip())
                except:
                    pass
            elif 'baseline host access lat:' in line:
                try:
                    result['baseline_host_access_lat'] = float(line.split(':')[-1].strip())
                except:
                    pass
            elif 'best_result:' in line:
                try:
                    result['best_cycles'] = float(line.split(':')[-1].strip())
                except:
                    pass
            elif 'best compute lat:' in line:
                try:
                    result['best_compute_lat'] = float(line.split(':')[-1].strip())
                except:
                    pass
            elif 'best row change lat:' in line:
                try:
                    result['best_row_change_lat'] = float(line.split(':')[-1].strip())
                except:
                    pass
            elif 'best host access lat:' in line:
                try:
                    result['best_host_access_lat'] = float(line.split(':')[-1].strip())
                except:
                    pass
            elif 'speedup:' in line:
                try:
                    result['speedup'] = float(line.split(':')[-1].strip())
                except:
                    pass
            elif 'best_design:' in line:
                result['best_design'] = line.split(':', 1)[-1].strip()
            elif 'baseline strategy:' in line:
                result['baseline_strategy'] = line.split(':', 1)[-1].strip()
        
        csv_path = os.path.join(OUTPUT_DIR, 'demo', 'csv', '_gui_demo.csv')
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 12 and row[0].strip() == 'gui_demo':
                            result['optimal_lat'] = int(row[2])
                            result['optimal_cmd'] = int(row[3])
                            result['optimal_pu_dram'] = int(row[4])
                            result['optimal_host_dram'] = int(row[5])
                            result['optimal_row_change'] = int(row[6])
                            result['baseline_lat'] = int(row[7])
                            result['baseline_cmd'] = int(row[8])
                            result['baseline_pu_dram'] = int(row[9])
                            result['baseline_host_dram'] = int(row[10])
                            result['baseline_row_change'] = int(row[11])
                            break
            except Exception as e:
                pass
        
        return result
    
    def stop(self):
        """Stop the running compilation"""
        self.should_stop = True
        if self.process:
            try:
                self.process.terminate()
            except:
                pass

# Global runner instance
compilation_runner = CompilationRunner()

# Global config files (loaded once)
_config_files = None

def get_config_files():
    global _config_files
    if _config_files is None:
        _config_files = load_config_files()
    return _config_files

# ============================================================================
# Gradio Interface
# ============================================================================

def sort_workload_keys(keys, workloads=None):
    """Sort workload keys with intelligent ordering for model sizes (e.g., 7B < 13B < 34B)
    
    Order: decode before prefill, preserve original file order within same file
    """
    import re
    
    def extract_sort_key(name):
        prefix_match = re.match(r'\[([^\]]+)\]', name)
        prefix = prefix_match.group(1) if prefix_match else name
        
        family_match = re.match(r'([a-zA-Z]+)', prefix)
        family = family_match.group(1).lower() if family_match else prefix.lower()
        
        size_match = re.search(r'_(\d+)B', prefix, re.IGNORECASE)
        model_size = int(size_match.group(1)) if size_match else 0
        
        phase = 0
        if 'decode' in prefix.lower():
            phase = 1
        elif 'prefill' in prefix.lower():
            phase = 2
        
        if workloads and name in workloads:
            file_order = workloads[name].get('file_order', 0)
        else:
            file_order = 0
        
        return (family, model_size, phase, file_order)
    
    return sorted(keys, key=extract_sort_key)

def create_interface():
    """Create the Gradio interface"""
    
    typical_workloads = load_typical_workloads()
    config_files = get_config_files()
    
    workload_choices = ['Custom'] + sort_workload_keys(list(typical_workloads.keys()), typical_workloads)
    arch_choices = list(ARCHITECTURE_MAP.keys())
    
    with gr.Blocks(title="UniNDP Compiler Demo") as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 2.2rem;">🚀 UniNDP Compiler Demo</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 10px; font-size: 1.1rem;">Matrix Multiplication Compilation for Near-Data Processing</p>
        </div>
        """)
        
        with gr.Row(equal_height=False):
            # Left Column: Input Panel
            with gr.Column(scale=1, min_width=350):
                gr.HTML('<div class="panel-header">📥 Input Configuration</div>')
                
                with gr.Accordion("Operator Configuration", open=True):
                    workload_dropdown = gr.Dropdown(
                        choices=workload_choices,
                        value='Custom',
                        label="Preset Workload (MM only)",
                        info="Select a typical MM workload or customize"
                    )
                    
                    with gr.Row():
                        m_input = gr.Number(value=1, label="M (Input)", precision=0, minimum=1)
                        k_input = gr.Number(value=4096, label="K (Reduce)", precision=0, minimum=1)
                        n_input = gr.Number(value=4096, label="N (Output)", precision=0, minimum=1)
                    
                    mm_viz = gr.HTML(value=generate_mm_svg(1, 4096, 4096, "Matrix Multiplication"))
                
                with gr.Accordion("Hardware Configuration", open=True):
                    arch_dropdown = gr.Dropdown(
                        choices=arch_choices,
                        value='UPMEM',
                        label="Architecture"
                    )
                    
                    hw_viz = gr.HTML(label="Hardware Architecture")
                    
                    with gr.Accordion("Advanced HW Options", open=True):
                        data_precision = gr.Number(value=16, label="Data Precision (bits)", precision=0, minimum=1)
                        pu_inbuf = gr.Number(value=0, label="PU Input Buffer (bits)", precision=0, minimum=0, visible=False)
                        pu_outbuf = gr.Number(value=0, label="PU Output Buffer (bits)", precision=0, minimum=0, visible=False)
                        pu_slowdown = gr.Number(value=1, label="PU Slowdown Factor", precision=0, minimum=1)
            
            # Middle Column: Search Config & Progress
            with gr.Column(scale=1, min_width=350):
                gr.HTML('<div class="panel-header">⚙️ Search Configuration</div>')
                
                with gr.Accordion("Search Options", open=True):
                    po2_check = gr.Checkbox(value=False, label="Power of 2 Partitioning (-P)")
                    quick_search = gr.Checkbox(value=True, label="Quick Search Mode (-Q)")
                    
                    with gr.Row():
                        topk_input = gr.Number(value=30, label="Top-K", precision=0, minimum=1)
                        cmdthre_input = gr.Number(value=3.0, label="Cmd Threshold", minimum=1.0)
                
                cmd_display = gr.Textbox(
                    label="Command Line",
                    interactive=False,
                    lines=2,
                    value="python compile.py -A upmem -W mm -S 1 4096 4096 1 -Q -K 30 -T 3.0"
                )
                
                with gr.Row():
                    run_btn = gr.Button("🚀 Start Compilation", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")
                
                gr.HTML('<div class="panel-header" style="margin-top: 15px;">📊 Compilation Progress</div>')
                
                progress_output = gr.Textbox(
                    label="",
                    lines=15,
                    max_lines=20,
                    interactive=False,
                    show_label=False
                )
                
                status_indicator = gr.HTML(
                    '<div style="text-align: center; padding: 8px; background: #e2e8f0; border-radius: 6px; font-size: 0.9rem;">Ready to compile</div>'
                )
            
            # Right Column: Results
            with gr.Column(scale=1, min_width=400):
                gr.HTML('<div class="panel-header">🎯 Optimization Results</div>')
                
                partition_viz = gr.HTML(
                    value='<div style="text-align: center; padding: 40px; color: #718096; background: #f7fafc; border-radius: 8px;">Run compilation to see partition strategy</div>'
                )
                
                dram_mapping_viz = gr.HTML(
                    value='<div style="text-align: center; padding: 40px; color: #718096; background: #f7fafc; border-radius: 8px;">Run compilation to see DRAM mapping</div>'
                )
                
                gr.HTML('<div class="panel-header">📈 Performance</div>')
                
                performance_viz = gr.HTML(
                    value='<div style="text-align: center; padding: 40px; color: #718096; background: #f7fafc; border-radius: 8px;">Run compilation to see performance results</div>'
                )
                
                with gr.Row():
                    baseline_cycles_display = gr.Number(label="Baseline Cycles", interactive=False, value=0)
                    best_cycles_display = gr.Number(label="Optimized Cycles", interactive=False, value=0)
        
        # Event Handlers
        def update_workload_from_preset(preset):
            if preset == 'Custom' or preset not in typical_workloads:
                return gr.update(), gr.update(), gr.update()
            
            wl = typical_workloads[preset]
            return (
                wl['sizes'][0],
                wl['sizes'][1],
                wl['sizes'][2]
            )
        
        def update_mm_visualization(m, k, n):
            m = int(m) if m is not None else 1
            k = int(k) if k is not None else 4096
            n = int(n) if n is not None else 4096
            return generate_mm_svg(m, k, n, "Matrix Multiplication")
        
        def update_command_line(arch, m, k, n, po2, quick, topk, cmdthre, datapr, puinbuf, puoutbuf, puslowdown):
            """Generate the command line string"""
            arch_key = ARCHITECTURE_MAP.get(arch, 'upmem')
            m = int(m) if m else 1
            k = int(k) if k else 4096
            n = int(n) if n else 4096
            
            cmd = f"python compile.py -A {arch_key} -W mm -S {m} {k} {n} 1"
            
            if po2:
                cmd += " -P"
            if quick:
                cmd += " -Q"
            
            topk_val = int(topk) if topk else 30
            cmdthre_val = float(cmdthre) if cmdthre else 3.0
            cmd += f" -K {topk_val} -T {cmdthre_val}"
            
            datapr_val = int(datapr) if datapr else 16
            puinbuf_val = int(puinbuf) if puinbuf else 0
            puoutbuf_val = int(puoutbuf) if puoutbuf else 0
            puslowdown_val = int(puslowdown) if puslowdown else 1
            
            if datapr_val != 16:
                cmd += f" -D {datapr_val}"
            if puinbuf_val > 0:
                cmd += f" -IB {puinbuf_val}"
            if puoutbuf_val > 0:
                cmd += f" -OB {puoutbuf_val}"
            if puslowdown_val > 1:
                cmd += f" -PS {puslowdown_val}"
            
            return cmd
        
        def update_hw_visualization(arch):
            config_file = get_config_for_architecture(arch)
            configs = get_config_files()
            if config_file in configs:
                return generate_hardware_html(configs[config_file], arch)
            return '<div style="text-align: center; padding: 40px; color: #718096;">Configuration not found</div>'
        
        def get_hw_defaults(arch):
            """Get default HW parameters based on architecture"""
            config_file = get_config_for_architecture(arch)
            configs = get_config_files()
            if config_file in configs:
                config = configs[config_file]
                data_pr = config.get('data_pr', 16)
                de_pu_inbuf = config.get('de_pu_inbuf', 0)
                de_pu_bf = config.get('de_pu_bf', 0)
                if 'dimmining' in arch.lower():
                    ra_pu_inbuf = config.get('ra_pu_inbuf', 0)
                    ra_pu_outbuf = config.get('ra_pu_outbuf', 0)
                    return data_pr, ra_pu_inbuf, ra_pu_outbuf
                if 'aim' in arch.lower():
                    de_gb = config.get('de_gb', 0)
                    if de_pu_inbuf == 0 and de_gb > 0:
                        return data_pr, de_gb, de_gb
                return data_pr, de_pu_inbuf, de_pu_bf
            return 16, 0, 0
        
        def run_compilation_streaming(arch, m, k, n, po2, quick, topk, cmdthre, datapr, puinbuf, puoutbuf, puslowdown):
            """Generator function for streaming compilation output"""
            
            options = {
                'po2': po2,
                'quicksearch': quick,
                'allow_under_utilize': False,
                'topk': int(topk) if topk else 30,
                'cmdthre': float(cmdthre) if cmdthre else 3.0,
                'datapr': int(datapr) if datapr else 16,
                'puinbuf': int(puinbuf) if puinbuf else 0,
                'puoutbuf': int(puoutbuf) if puoutbuf else 0,
                'puslowdown': int(puslowdown) if puslowdown else 1
            }
            
            sizes = [
                int(m) if m else 1,
                int(k) if k else 4096,
                int(n) if n else 4096,
                1
            ]
            
            status_running = '<div style="text-align: center; padding: 8px; background: #bee3f8; border-radius: 6px; color: #2b6cb0; font-size: 0.9rem;">⏳ Compiling...</div>'
            compiling_placeholder = '<div style="text-align: center; padding: 40px; color: #718096; background: #f7fafc; border-radius: 8px;">Compiling...</div>'
            
            yield (
                "Starting compilation...\n",
                status_running,
                compiling_placeholder,
                compiling_placeholder,
                compiling_placeholder,
                0,
                0
            )
            
            last_output = ""
            for output in compilation_runner.run_compilation_stream(arch, 'mm', sizes, options):
                last_output = output
                yield (
                    output,
                    status_running,
                    compiling_placeholder,
                    compiling_placeholder,
                    compiling_placeholder,
                    0,
                    0
            )
            
            result = compilation_runner.result
            
            if result:
                baseline_cycles = result.get('baseline_cycles', result.get('baseline_lat', 0))
                best_cycles = result.get('best_cycles', result.get('optimal_lat', 0))
                speedup = result.get('speedup', 1.0)
                
                perf_html = generate_performance_html(result)
                
                partition_html = '<div style="text-align: center; padding: 40px; color: #718096; background: #f7fafc; border-radius: 8px;">Partition data not available</div>'
                dram_mapping_html = '<div style="text-align: center; padding: 40px; color: #718096; background: #f7fafc; border-radius: 8px;">DRAM mapping data not available</div>'
                
                try:
                    best_design_str = result.get('best_design', '')
                    if best_design_str:
                        config_file = get_config_for_architecture(arch)
                        configs = get_config_files()
                        config = configs.get(config_file, {})
                        partition_html = generate_detailed_partition_html(
                            best_design_str,
                            sizes,
                            config,
                            arch
                        )
                        dram_mapping_html = generate_dram_mapping_html(
                            best_design_str,
                            sizes,
                            config,
                            arch
                        )
                except Exception as e:
                    partition_html = f'<div style="color: #e53e3e; padding: 15px;">Error parsing partition: {str(e)}</div>'
                
                status_html = '<div style="text-align: center; padding: 8px; background: #c6f6d5; border-radius: 6px; color: #276749; font-size: 0.9rem;">✅ Compilation completed</div>'
                
                yield (
                    last_output,
                    status_html,
                    partition_html,
                    dram_mapping_html,
                    perf_html,
                    baseline_cycles,
                    best_cycles
                )
            else:
                no_data = '<div style="text-align: center; padding: 40px; color: #718096; background: #f7fafc; border-radius: 8px;">No data</div>'
                yield (
                    last_output,
                    '<div style="text-align: center; padding: 8px; background: #fed7d7; border-radius: 6px; color: #c53030; font-size: 0.9rem;">❌ Compilation failed</div>',
                    no_data,
                    no_data,
                    no_data,
                    0,
                    0
                )
        
        def stop_compilation():
            compilation_runner.stop()
            return '<div style="text-align: center; padding: 8px; background: #feebc8; border-radius: 6px; color: #c05621; font-size: 0.9rem;">⚠️ Stopped by user</div>'
        
        # Wire up events
        workload_dropdown.change(
            update_workload_from_preset,
            inputs=[workload_dropdown],
            outputs=[m_input, k_input, n_input]
        )
        
        for input_component in [m_input, k_input, n_input]:
            input_component.change(
                update_mm_visualization,
                inputs=[m_input, k_input, n_input],
                outputs=[mm_viz]
            )
        
        def on_arch_change(arch):
            hw_html = update_hw_visualization(arch)
            data_pr, inbuf, outbuf = get_hw_defaults(arch)
            is_hbm_pim = 'hbm' in arch.lower()
            return (
                hw_html, 
                data_pr, 
                gr.update(value=inbuf, visible=is_hbm_pim),
                gr.update(value=outbuf, visible=is_hbm_pim)
            )
        
        arch_dropdown.change(
            on_arch_change,
            inputs=[arch_dropdown],
            outputs=[hw_viz, data_precision, pu_inbuf, pu_outbuf]
        )
        
        cmd_inputs = [arch_dropdown, m_input, k_input, n_input, po2_check, quick_search, 
                      topk_input, cmdthre_input, data_precision, pu_inbuf, pu_outbuf, pu_slowdown]
        for input_component in cmd_inputs:
            input_component.change(
                update_command_line,
                inputs=cmd_inputs,
                outputs=[cmd_display]
        )
        
        run_btn.click(
            run_compilation_streaming,
            inputs=[
                arch_dropdown, m_input, k_input, n_input,
                po2_check, quick_search, topk_input, cmdthre_input,
                data_precision, pu_inbuf, pu_outbuf, pu_slowdown
            ],
            outputs=[
                progress_output, status_indicator, partition_viz, dram_mapping_viz, performance_viz,
                baseline_cycles_display, best_cycles_display
            ]
        )
        
        stop_btn.click(
            stop_compilation,
            outputs=[status_indicator]
        )
        
        # Initialize visualizations on load
        def init_visualizations():
            data_pr, inbuf, outbuf = get_hw_defaults('UPMEM')
            return (
                generate_mm_svg(1, 4096, 4096, "Matrix Multiplication"),
                update_hw_visualization('UPMEM'),
                update_command_line('UPMEM', 1, 4096, 4096, False, True, 30, 3.0, data_pr, 0, 0, 1),
                data_pr,
                gr.update(value=0, visible=False),
                gr.update(value=0, visible=False)
            )
        
        demo.load(
            init_visualizations,
            outputs=[mm_viz, hw_viz, cmd_display, data_precision, pu_inbuf, pu_outbuf]
        )
    
    return demo

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    print("Starting UniNDP GUI Demo...")
    print("Loading configurations...")
    
    demo = create_interface()
    
    print("\n" + "="*60)
    print("UniNDP Compiler GUI Demo")
    print("="*60)
    print("Opening browser at http://127.0.0.1:7860")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    import socket
    
    def find_free_port(start_port=7860, max_attempts=10):
        for port in range(start_port, start_port + max_attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('', port))
                sock.close()
                return port
            except OSError:
                continue
        return start_port + max_attempts
    
    port = find_free_port()
    print(f"Starting server on port {port}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        inbrowser=False,
        css=CUSTOM_CSS
    )
