#!/usr/bin/env python3
"""
Security & Defense Division Architecture
========================================

Personal AI Security Guardian - Lives on your devices to protect you.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
import json

@dataclass
class SecurityLabSpec:
    """Specification for a security-focused laboratory"""
    name: str
    tier: int  # 1=Core Detection, 2=Threat Analysis, 3=Active Defense, 4=Predictive
    parent_labs: List[str]
    focus_area: str
    key_capabilities: List[str]
    detection_methods: List[str]
    response_actions: List[str]
    resource_usage: Dict[str, float]  # CPU, RAM, Battery impact
    privacy_level: str  # "full_local", "hybrid", "cloud_optional"
    evolution_generations: int

class PersonalSecurityDivision:
    """Architecture for personal device security AI"""
    
    def __init__(self):
        self.labs = {}
        self._build_security_hierarchy()
    
    def _build_security_hierarchy(self):
        """Build the security laboratory hierarchy"""
        
        # =========================================
        # TIER 1: CORE DETECTION ENGINES
        # =========================================
        
        self.labs["anomaly_baseline"] = SecurityLabSpec(
            name="Personal Behavior Baseline",
            tier=1,
            parent_labs=["Pattern", "Adaptation"],
            focus_area="Learn your normal digital behavior",
            key_capabilities=[
                "App usage pattern learning",
                "Network behavior profiling",
                "File access pattern mapping",
                "Communication pattern analysis",
                "Location pattern understanding"
            ],
            detection_methods=[
                "Statistical deviation detection",
                "Time-series anomaly detection",
                "Clustering-based outliers",
                "Markov chain disruptions"
            ],
            response_actions=[
                "Silent logging",
                "Baseline updates",
                "Anomaly scoring"
            ],
            resource_usage={"cpu": 0.05, "ram": 0.1, "battery": 0.02},
            privacy_level="full_local",
            evolution_generations=1000
        )
        
        self.labs["network_guardian"] = SecurityLabSpec(
            name="Network Traffic Guardian",
            tier=1,
            parent_labs=["Pattern", "Prediction"],
            focus_area="Monitor all network connections",
            key_capabilities=[
                "Real-time packet inspection",
                "DNS query analysis",
                "SSL certificate validation",
                "Data exfiltration detection",
                "Suspicious connection identification"
            ],
            detection_methods=[
                "DPI without decryption",
                "Traffic pattern analysis",
                "Destination reputation check",
                "Volume anomaly detection",
                "Protocol abuse detection"
            ],
            response_actions=[
                "Block suspicious connections",
                "Alert on data leaks",
                "Quarantine apps",
                "VPN auto-activation"
            ],
            resource_usage={"cpu": 0.15, "ram": 0.2, "battery": 0.05},
            privacy_level="full_local",
            evolution_generations=1500
        )
        
        self.labs["app_sentinel"] = SecurityLabSpec(
            name="Application Behavior Sentinel",
            tier=1,
            parent_labs=["Pattern", "Cognitive"],
            focus_area="Monitor app behaviors and permissions",
            key_capabilities=[
                "Permission abuse detection",
                "Background activity monitoring",
                "Resource usage profiling",
                "Inter-app communication tracking",
                "Code injection detection"
            ],
            detection_methods=[
                "API call monitoring",
                "Memory access patterns",
                "File system monitoring",
                "Sensor access tracking",
                "Battery drain analysis"
            ],
            response_actions=[
                "Revoke permissions",
                "Freeze suspicious apps",
                "Sandbox isolation",
                "Uninstall recommendation"
            ],
            resource_usage={"cpu": 0.1, "ram": 0.15, "battery": 0.03},
            privacy_level="full_local",
            evolution_generations=1200
        )
        
        # =========================================
        # TIER 2: THREAT ANALYSIS SPECIALISTS
        # =========================================
        
        self.labs["malware_hunter"] = SecurityLabSpec(
            name="Advanced Malware Hunter",
            tier=2,
            parent_labs=["app_sentinel", "anomaly_baseline"],
            focus_area="Detect sophisticated malware",
            key_capabilities=[
                "Behavioral malware detection",
                "Polymorphic code detection",
                "Rootkit discovery",
                "Zero-day behavior patterns",
                "Fileless malware detection"
            ],
            detection_methods=[
                "Heuristic analysis",
                "Machine learning signatures",
                "Sandbox detonation",
                "Memory forensics",
                "Cross-reference with patterns"
            ],
            response_actions=[
                "Immediate isolation",
                "System restore points",
                "Evidence collection",
                "Threat removal"
            ],
            resource_usage={"cpu": 0.2, "ram": 0.25, "battery": 0.08},
            privacy_level="hybrid",
            evolution_generations=2000
        )
        
        self.labs["phishing_shield"] = SecurityLabSpec(
            name="Anti-Phishing Shield",
            tier=2,
            parent_labs=["network_guardian", "Cognitive"],
            focus_area="Protect against phishing and social engineering",
            key_capabilities=[
                "URL reputation analysis",
                "Visual similarity detection",
                "Email header analysis",
                "Social engineering detection",
                "Credential theft prevention"
            ],
            detection_methods=[
                "Domain similarity scoring",
                "Content analysis",
                "Sender verification",
                "Link target analysis",
                "Image recognition"
            ],
            response_actions=[
                "Block phishing sites",
                "Warn before submission",
                "Password manager integration",
                "Report to authorities"
            ],
            resource_usage={"cpu": 0.1, "ram": 0.1, "battery": 0.02},
            privacy_level="hybrid",
            evolution_generations=1000
        )
        
        self.labs["privacy_guardian"] = SecurityLabSpec(
            name="Privacy Protection Guardian",
            tier=2,
            parent_labs=["app_sentinel", "network_guardian"],
            focus_area="Protect personal data and privacy",
            key_capabilities=[
                "Data leak prevention",
                "Camera/mic access monitoring",
                "Location tracking detection",
                "Contact list protection",
                "Clipboard monitoring"
            ],
            detection_methods=[
                "Sensitive data classification",
                "Unauthorized access detection",
                "Cross-app data sharing",
                "Background recording detection",
                "Privacy policy analysis"
            ],
            response_actions=[
                "Block data transmission",
                "Fake data injection",
                "Access notifications",
                "Privacy report generation"
            ],
            resource_usage={"cpu": 0.12, "ram": 0.18, "battery": 0.04},
            privacy_level="full_local",
            evolution_generations=1500
        )
        
        self.labs["crypto_defender"] = SecurityLabSpec(
            name="Cryptocurrency Defender",
            tier=2,
            parent_labs=["network_guardian", "app_sentinel"],
            focus_area="Protect crypto wallets and transactions",
            key_capabilities=[
                "Wallet security monitoring",
                "Transaction verification",
                "Clipboard hijack prevention",
                "Fake wallet detection",
                "Private key protection"
            ],
            detection_methods=[
                "Address validation",
                "Transaction anomaly detection",
                "Clipboard manipulation detection",
                "Screen recording detection",
                "Keylogger detection"
            ],
            response_actions=[
                "Transaction blocking",
                "Wallet isolation",
                "2FA enforcement",
                "Cold storage recommendation"
            ],
            resource_usage={"cpu": 0.08, "ram": 0.1, "battery": 0.02},
            privacy_level="full_local",
            evolution_generations=1200
        )
        
        # =========================================
        # TIER 3: ACTIVE DEFENSE SYSTEMS
        # =========================================
        
        self.labs["deception_system"] = SecurityLabSpec(
            name="Deception & Honeypot System",
            tier=3,
            parent_labs=["malware_hunter", "privacy_guardian"],
            focus_area="Active deception against attackers",
            key_capabilities=[
                "Fake data generation",
                "Honeypot services",
                "Attacker profiling",
                "False trail creation",
                "Time-wasting tactics"
            ],
            detection_methods=[
                "Honeypot access detection",
                "Attacker fingerprinting",
                "Attack pattern learning",
                "Tool identification",
                "Attribution analysis"
            ],
            response_actions=[
                "Deploy decoys",
                "Waste attacker time",
                "Collect intelligence",
                "Legal evidence gathering"
            ],
            resource_usage={"cpu": 0.15, "ram": 0.2, "battery": 0.05},
            privacy_level="full_local",
            evolution_generations=1800
        )
        
        self.labs["auto_hardening"] = SecurityLabSpec(
            name="Automatic System Hardening",
            tier=3,
            parent_labs=["anomaly_baseline", "malware_hunter"],
            focus_area="Continuously harden system security",
            key_capabilities=[
                "Configuration optimization",
                "Vulnerability patching",
                "Permission minimization",
                "Attack surface reduction",
                "Security policy enforcement"
            ],
            detection_methods=[
                "Vulnerability scanning",
                "Configuration drift detection",
                "Exploit attempt detection",
                "Privilege escalation detection",
                "Compliance checking"
            ],
            response_actions=[
                "Auto-patch critical vulns",
                "Disable risky features",
                "Enforce security policies",
                "Rollback bad changes"
            ],
            resource_usage={"cpu": 0.1, "ram": 0.15, "battery": 0.03},
            privacy_level="full_local",
            evolution_generations=1500
        )
        
        self.labs["incident_responder"] = SecurityLabSpec(
            name="Automated Incident Response",
            tier=3,
            parent_labs=["malware_hunter", "network_guardian"],
            focus_area="Respond to security incidents automatically",
            key_capabilities=[
                "Incident classification",
                "Automated containment",
                "Evidence preservation",
                "System recovery",
                "Post-incident analysis"
            ],
            detection_methods=[
                "Attack chain analysis",
                "Impact assessment",
                "Lateral movement detection",
                "Data breach detection",
                "System compromise indicators"
            ],
            response_actions=[
                "Isolate infected systems",
                "Kill malicious processes",
                "Restore from backups",
                "Generate incident reports"
            ],
            resource_usage={"cpu": 0.25, "ram": 0.3, "battery": 0.1},
            privacy_level="full_local",
            evolution_generations=2000
        )
        
        # =========================================
        # TIER 4: PREDICTIVE SECURITY
        # =========================================
        
        self.labs["threat_predictor"] = SecurityLabSpec(
            name="Predictive Threat Intelligence",
            tier=4,
            parent_labs=["incident_responder", "Prediction"],
            focus_area="Predict attacks before they happen",
            key_capabilities=[
                "Attack trend prediction",
                "Vulnerability forecasting",
                "Threat actor profiling",
                "Zero-day prediction",
                "Campaign detection"
            ],
            detection_methods=[
                "Pattern extrapolation",
                "Threat intelligence fusion",
                "Behavioral prediction",
                "Attack surface modeling",
                "Risk scoring"
            ],
            response_actions=[
                "Preemptive hardening",
                "Early warning alerts",
                "Proactive patching",
                "Behavior modification"
            ],
            resource_usage={"cpu": 0.2, "ram": 0.25, "battery": 0.08},
            privacy_level="hybrid",
            evolution_generations=2500
        )
        
        self.labs["social_shield"] = SecurityLabSpec(
            name="Social Engineering Shield",
            tier=4,
            parent_labs=["phishing_shield", "Cognitive"],
            focus_area="Protect against human-targeted attacks",
            key_capabilities=[
                "Scam pattern recognition",
                "Deepfake detection",
                "Impersonation detection",
                "Manipulation detection",
                "Trust scoring"
            ],
            detection_methods=[
                "Communication analysis",
                "Behavioral biometrics",
                "Voice analysis",
                "Writing style analysis",
                "Social graph analysis"
            ],
            response_actions=[
                "Real-time warnings",
                "Communication filtering",
                "Identity verification",
                "Trust recommendations"
            ],
            resource_usage={"cpu": 0.18, "ram": 0.2, "battery": 0.06},
            privacy_level="full_local",
            evolution_generations=2000
        )
    
    def get_minimal_protection_stack(self) -> List[str]:
        """Get minimal set of labs for basic protection"""
        return [
            "anomaly_baseline",
            "network_guardian", 
            "app_sentinel",
            "phishing_shield"
        ]
    
    def get_comprehensive_stack(self) -> List[str]:
        """Get comprehensive protection stack"""
        return [lab for lab in self.labs.keys()]
    
    def calculate_resource_usage(self, active_labs: List[str]) -> Dict[str, float]:
        """Calculate total resource usage for active labs"""
        total = {"cpu": 0.0, "ram": 0.0, "battery": 0.0}
        for lab_name in active_labs:
            if lab_name in self.labs:
                lab = self.labs[lab_name]
                for resource, usage in lab.resource_usage.items():
                    total[resource] += usage
        return total

def create_personal_guardian_system():
    """Create a personal security guardian implementation"""
    
    implementation = '''#!/usr/bin/env python3
"""
Personal Security Guardian
=========================

AI that lives on your device to protect you.
"""

import asyncio
import time
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os

@dataclass
class ThreatAlert:
    severity: str  # low, medium, high, critical
    source: str
    description: str
    timestamp: datetime
    actions_taken: List[str]
    evidence: Dict

class PersonalSecurityGuardian:
    def __init__(self, protection_level: str = "balanced"):
        self.protection_level = protection_level
        self.active_labs = self._select_labs(protection_level)
        self.threat_history = []
        self.baseline_data = {}
        self.is_learning = True
        
    def _select_labs(self, level: str) -> List[str]:
        """Select which security labs to activate"""
        if level == "minimal":
            return ["anomaly_baseline", "network_guardian"]
        elif level == "balanced":
            return [
                "anomaly_baseline",
                "network_guardian",
                "app_sentinel",
                "phishing_shield",
                "privacy_guardian"
            ]
        elif level == "maximum":
            return [
                "anomaly_baseline",
                "network_guardian",
                "app_sentinel",
                "malware_hunter",
                "phishing_shield",
                "privacy_guardian",
                "crypto_defender",
                "incident_responder"
            ]
        else:  # paranoid
            return "all"  # Activate everything
    
    async def start_protection(self):
        """Start the security guardian"""
        print("🛡️ Personal Security Guardian Starting...")
        print(f"Protection Level: {self.protection_level}")
        print(f"Active Modules: {len(self.active_labs)}")
        
        # Start learning phase
        if self.is_learning:
            await self.learn_baseline()
        
        # Start protection tasks
        tasks = [
            asyncio.create_task(self.monitor_network()),
            asyncio.create_task(self.monitor_apps()),
            asyncio.create_task(self.monitor_system()),
            asyncio.create_task(self.analyze_threats()),
            asyncio.create_task(self.update_intelligence())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\\n🛑 Guardian shutting down...")
    
    async def learn_baseline(self):
        """Learn normal behavior patterns"""
        print("📊 Learning your normal behavior patterns...")
        
        baseline_tasks = {
            "network_patterns": self.learn_network_baseline(),
            "app_usage": self.learn_app_baseline(),
            "file_access": self.learn_file_baseline(),
            "system_behavior": self.learn_system_baseline()
        }
        
        for task_name, task in baseline_tasks.items():
            print(f"  Learning {task_name}...")
            await task
        
        self.is_learning = False
        print("✅ Baseline established!")
    
    async def monitor_network(self):
        """Monitor network connections"""
        while True:
            # Check all active connections
            connections = await self.get_network_connections()
            
            for conn in connections:
                # Check against baseline
                if self.is_anomalous_connection(conn):
                    threat = ThreatAlert(
                        severity="medium",
                        source="network_guardian",
                        description=f"Unusual connection to {conn['dest']}",
                        timestamp=datetime.now(),
                        actions_taken=["logged", "monitoring"],
                        evidence={"connection": conn}
                    )
                    
                    # Decide action based on severity
                    if conn.get("risk_score", 0) > 0.8:
                        await self.block_connection(conn)
                        threat.actions_taken.append("blocked")
                        threat.severity = "high"
                    
                    self.threat_history.append(threat)
                    await self.notify_user(threat)
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def monitor_apps(self):
        """Monitor application behavior"""
        while True:
            # Get running apps
            apps = await self.get_running_apps()
            
            for app in apps:
                # Check permissions
                if self.has_excessive_permissions(app):
                    await self.suggest_permission_reduction(app)
                
                # Check behavior
                behavior = await self.analyze_app_behavior(app)
                if behavior.get("suspicious", False):
                    threat = ThreatAlert(
                        severity="medium",
                        source="app_sentinel",
                        description=f"{app['name']} showing suspicious behavior",
                        timestamp=datetime.now(),
                        actions_taken=["monitoring"],
                        evidence=behavior
                    )
                    
                    if behavior.get("malware_probability", 0) > 0.7:
                        await self.quarantine_app(app)
                        threat.actions_taken.append("quarantined")
                        threat.severity = "critical"
                    
                    self.threat_history.append(threat)
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def detect_active_threats(self):
        """Detect active threats in real-time"""
        detectors = {
            "malware": self.detect_malware,
            "phishing": self.detect_phishing,
            "data_leak": self.detect_data_leak,
            "crypto_theft": self.detect_crypto_theft,
            "privacy_violation": self.detect_privacy_violation
        }
        
        active_threats = []
        for threat_type, detector in detectors.items():
            if threat_type in self.active_labs or "all" in self.active_labs:
                threats = await detector()
                active_threats.extend(threats)
        
        return active_threats
    
    async def respond_to_threat(self, threat: ThreatAlert):
        """Automated threat response"""
        responses = {
            "critical": [
                self.isolate_system,
                self.kill_malicious_processes,
                self.notify_emergency
            ],
            "high": [
                self.block_threat_source,
                self.enhance_monitoring,
                self.notify_user
            ],
            "medium": [
                self.log_threat,
                self.monitor_closely,
                self.suggest_action
            ],
            "low": [
                self.log_threat,
                self.update_baseline
            ]
        }
        
        for response_action in responses.get(threat.severity, []):
            await response_action(threat)
    
    def generate_security_report(self) -> Dict:
        """Generate security status report"""
        return {
            "status": "protected",
            "protection_level": self.protection_level,
            "active_modules": len(self.active_labs),
            "threats_blocked_today": len([
                t for t in self.threat_history 
                if t.timestamp.date() == datetime.now().date()
            ]),
            "current_risk_level": self.calculate_risk_level(),
            "recommendations": self.get_security_recommendations()
        }

# Usage Example
async def main():
    # Create guardian with balanced protection
    guardian = PersonalSecurityGuardian(protection_level="balanced")
    
    # Start protection
    await guardian.start_protection()

if __name__ == "__main__":
    print("🚀 Personal Security Guardian v1.0")
    print("================================")
    asyncio.run(main())
'''
    
    return implementation

def create_mobile_integration():
    """Create mobile-specific integration"""
    
    mobile_code = '''#!/usr/bin/env python3
"""
Mobile Security Guardian
=======================

Lightweight version for smartphones.
"""

class MobileSecurityGuardian:
    def __init__(self):
        self.battery_saver_mode = True
        self.background_scanning = True
        self.real_time_protection = True
        
    def get_optimal_config(self, device_type: str) -> Dict:
        """Get optimal configuration for device"""
        configs = {
            "flagship": {  # High-end phones
                "active_labs": [
                    "anomaly_baseline",
                    "network_guardian",
                    "app_sentinel",
                    "phishing_shield",
                    "privacy_guardian",
                    "malware_hunter"
                ],
                "scan_frequency": "real_time",
                "battery_impact": "moderate"
            },
            "midrange": {  # Mid-range phones
                "active_labs": [
                    "anomaly_baseline",
                    "network_guardian",
                    "app_sentinel",
                    "phishing_shield"
                ],
                "scan_frequency": "periodic",
                "battery_impact": "low"
            },
            "budget": {  # Low-end phones
                "active_labs": [
                    "network_guardian",
                    "phishing_shield"
                ],
                "scan_frequency": "on_demand",
                "battery_impact": "minimal"
            }
        }
        return configs.get(device_type, configs["midrange"])
    
    def ios_specific_features(self):
        """iOS-specific security features"""
        return {
            "app_store_verification": True,
            "sandbox_monitoring": True,
            "jailbreak_detection": True,
            "certificate_pinning": True,
            "keychain_protection": True
        }
    
    def android_specific_features(self):
        """Android-specific security features"""
        return {
            "apk_scanning": True,
            "permission_management": True,
            "root_detection": True,
            "google_play_protect_integration": True,
            "sideload_protection": True
        }
    
    def quick_scan(self) -> Dict:
        """Perform quick security scan"""
        results = {
            "network_threats": self.scan_network_quick(),
            "app_risks": self.scan_apps_quick(),
            "privacy_issues": self.scan_privacy_quick(),
            "overall_score": 0
        }
        
        # Calculate overall security score
        threat_count = sum(len(v) for v in results.values() if isinstance(v, list))
        results["overall_score"] = max(0, 100 - (threat_count * 10))
        
        return results

# Desktop Integration
class DesktopSecurityGuardian:
    def __init__(self):
        self.full_system_access = True
        self.deep_scanning = True
        self.forensics_capable = True
    
    def windows_specific(self):
        return {
            "defender_integration": True,
            "registry_monitoring": True,
            "dll_injection_detection": True,
            "powershell_monitoring": True
        }
    
    def macos_specific(self):
        return {
            "gatekeeper_enhancement": True,
            "xprotect_integration": True,
            "kernel_extension_monitoring": True,
            "privacy_tcc_protection": True
        }
    
    def linux_specific(self):
        return {
            "selinux_integration": True,
            "kernel_module_monitoring": True,
            "package_verification": True,
            "sudo_monitoring": True
        }
'''
    
    return mobile_code

# Main execution
if __name__ == "__main__":
    division = PersonalSecurityDivision()
    
    print("="*80)
    print("🛡️ PERSONAL SECURITY DIVISION ARCHITECTURE")
    print("="*80)
    
    # Show tier distribution
    tier_summary = {1: [], 2: [], 3: [], 4: []}
    for lab_name, lab in division.labs.items():
        tier_summary[lab.tier].append(lab_name)
    
    print("\n📊 SECURITY LABORATORY TIERS:")
    tier_names = {
        1: "Core Detection",
        2: "Threat Analysis", 
        3: "Active Defense",
        4: "Predictive Security"
    }
    
    for tier, labs in tier_summary.items():
        print(f"\nTIER {tier} - {tier_names[tier]} ({len(labs)} labs):")
        for lab in labs:
            spec = division.labs[lab]
            print(f"  • {spec.name}")
            print(f"    Focus: {spec.focus_area}")
            print(f"    Resources: CPU {spec.resource_usage['cpu']*100:.0f}%, "
                  f"RAM {spec.resource_usage['ram']*100:.0f}%, "
                  f"Battery {spec.resource_usage['battery']*100:.0f}%")
    
    # Protection stacks
    print("\n🔰 PROTECTION STACKS:")
    
    minimal = division.get_minimal_protection_stack()
    print(f"\nMinimal Protection ({len(minimal)} labs):")
    resources = division.calculate_resource_usage(minimal)
    print(f"  Resources: CPU {resources['cpu']*100:.0f}%, "
          f"RAM {resources['ram']*100:.0f}%, "
          f"Battery {resources['battery']*100:.0f}%")
    for lab in minimal:
        print(f"  • {division.labs[lab].name}")
    
    comprehensive = division.get_comprehensive_stack()
    print(f"\nComprehensive Protection ({len(comprehensive)} labs):")
    resources = division.calculate_resource_usage(comprehensive)
    print(f"  Resources: CPU {resources['cpu']*100:.0f}%, "
          f"RAM {resources['ram']*100:.0f}%, "
          f"Battery {resources['battery']*100:.0f}%")
    
    # Privacy focus
    print("\n🔐 PRIVACY-FIRST APPROACH:")
    full_local = [lab for lab, spec in division.labs.items() 
                  if spec.privacy_level == "full_local"]
    print(f"{len(full_local)} labs work completely offline:")
    for lab in full_local[:5]:
        print(f"  • {division.labs[lab].name}")
    
    # Key capabilities
    print("\n💪 KEY PROTECTION CAPABILITIES:")
    capabilities = [
        "🔍 Real-time threat detection without cloud dependency",
        "🧠 Learn your personal usage patterns",
        "🛡️ Block attacks before damage occurs",
        "🔒 Protect privacy with local processing",
        "⚡ Minimal battery impact with smart scheduling",
        "🎯 Predictive threat prevention",
        "🪤 Active deception against attackers",
        "🔄 Automatic security hardening"
    ]
    for cap in capabilities:
        print(f"  {cap}")
    
    # Save implementation
    with open("personal_guardian.py", "w") as f:
        f.write(create_personal_guardian_system())
    
    with open("mobile_security.py", "w") as f:
        f.write(create_mobile_integration())
    
    print("\n📄 Implementation files created:")
    print("  • personal_guardian.py - Main security system")
    print("  • mobile_security.py - Mobile integration")