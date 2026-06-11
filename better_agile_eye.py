'''
Author: Alp Oran
Made to test some optimization routines I came up with for MCU's. 
Added some visualization just to make sure everything works
'''

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

#based on the Cordic algortihm, wanted to test out if I could make one without branching
def atan2Cordic(yVal, xVal):
    xVal = np.atleast_1d(xVal).astype(float)
    yVal = np.atleast_1d(yVal).astype(float)

    angTab = np.array(
        [
            0.7853981633974483,   # atan(1)
            0.4636476090008061,   # atan(0.5)
            0.24497866312686414,  # atan(0.25)
            0.12435499454676144,  # atan(0.125)
            0.0624188099922018,   # atan(0.0625)
            0.031239833430268277, # atan(0.03125)
            0.015623728620476831, # atan(0.015625)
            0.00781211796494941,  # atan(0.0078125)
        ]
    )
    
    lHP = xVal < 0

    xi = np.where(lHP, -xVal, xVal)
    yi = np.where(lHP, -yVal, yVal)

    bAng = np.zeros_like(xVal)
    bAng = np.where(lHP & (yVal >= 0), np.pi, bAng)
    bAng = np.where(lHP & (yVal < 0), -np.pi, bAng)

    bAng = np.where((xVal == 0) & (yVal > 0), np.pi / 2, bAng)
    bAng = np.where((xVal == 0) & (yVal < 0), -np.pi / 2, bAng)
    bAng = np.where((xVal == 0) & (yVal == 0), 0.0, bAng)

    for idx in range(len(angTab)):
        sig = np.where(yi >= 0, 1.0, -1.0)
        tanFact = 1.0 / (2**idx)

        nxtX = xi + sig * yi * tanFact
        nxtY = yi - sig * xi * tanFact

        xi = nxtX
        yi = nxtY

        bAng += sig * angTab[idx]

    return bAng if bAng.size > 1 else bAng[0]

# Standard quaternion multiplication
def qMult(pQ, qQ):
    pW, pX, pY, pZ = pQ[0], pQ[1], pQ[2], pQ[3]
    qW, qX, qY, qZ = qQ[0], qQ[1], qQ[2], qQ[3]
    
    mRes = np.vstack((
        pW*qW - pX*qX - pY*qY - pZ*qZ,
        pW*qX + pX*qW + pY*qZ - pZ*qY,
        pW*qY - pX*qZ + pY*qW + pZ*qX,
        pW*qZ + pX*qY - pY*qX + pZ*qW
    ))
    if mRes.shape[1] == 1 and pQ.ndim == 1 and qQ.ndim == 1:
        return mRes.ravel()
    return mRes

# More info in the attached doc. TL Dr; uses no trig calls but still computes the motor
def qExp(uAxis, theta, pThresh=-10):
    theta = np.asarray(theta, dtype=np.float64)
    uAxis = np.asarray(uAxis, dtype=np.float64)
    
    if theta.ndim == 0: theta = theta.reshape(1)
    if uAxis.ndim == 1: uAxis = uAxis.reshape(3, 1)
        
    _, bExp = np.frexp(theta)
    scStep = np.clip(bExp - pThresh, 0, None).astype(int)
    scTheta = np.ldexp(theta, -(scStep + 1))
    
    qScalar = 1.0 - scTheta**2 * 0.5
    qVector = uAxis * scTheta
    bQuat = np.vstack((qScalar, qVector))
    
    mSteps = int(np.max(scStep))
    for idx in range(mSteps):
        iMask = (idx < scStep).astype(float)
        expMask = np.expand_dims(iMask, axis=0)
        
        sqQuat = qMult(bQuat, bQuat)
        bQuat = (expMask * sqQuat) + ((1.0 - expMask) * bQuat)
        
    if bQuat.shape[1] == 1: return bQuat.ravel()
    return bQuat

# wrote a inverse function to make my life easier should I need it :)
def qInv(tQuat):
    if tQuat.ndim == 1: return np.array([tQuat[0], -tQuat[1], -tQuat[2], -tQuat[3]])
    return np.vstack((tQuat[0], -tQuat[1], -tQuat[2], -tQuat[3]))


def qRot(tQuat, tPt):
    tQuat = np.asarray(tQuat, dtype=np.float64)
    tPt = np.asarray(tPt, dtype=np.float64)
    isSingle = (tPt.ndim == 1)
    
    if isSingle: tPt = tPt.reshape(3, 1)
    if tQuat.ndim == 1: tQuat = tQuat.reshape(4, 1)
        
    pZeros = np.zeros((1, tPt.shape[1]), dtype=np.float64)
    pPt = np.vstack((pZeros, tPt)) 
    
    fStep = qMult(tQuat, pPt)
    iStep = qMult(fStep, qInv(tQuat))
    eRes = iStep[1:4, :]
    return eRes.ravel() if isSingle else eRes

#Parameters for the agile eye
alphaDeg, betaDeg = 80.0, 100.0  
alphaRad = alphaDeg * pi / 180.0
betaRad = betaDeg * pi / 180.0

cosAlpha, cosBeta = np.cos(alphaRad), np.cos(betaRad)

outRad, inRad = 20.0, 10.0   
zOff = outRad            

zAxis = np.array([0, 0, 1])
q120 = qExp(zAxis, 120.0 * pi / 180.0)

P1 = np.array([inRad, 0, 0])
P2 = qRot(q120, P1)
P3 = qRot(q120, P2)

kAng = np.arccos(zOff / outRad)
P12 = np.array([outRad * np.sin(kAng), 0, -zOff])
P22 = qRot(q120, P12)
P32 = qRot(q120, P22)

platHome = np.column_stack((P1, P2, P3))
baseHome = np.column_stack((P12, P22, P32))
bUnitVecs = baseHome / np.linalg.norm(baseHome, axis=0)

def computeSlerpArc(vStart, vEnd, absRad, pts=20):
    sUnit, eUnit = vStart / np.linalg.norm(vStart), vEnd / np.linalg.norm(vEnd)
    angSep = np.arccos(np.clip(np.dot(sUnit, eUnit), -1.0, 1.0))
    if angSep < 1e-5: return np.outer(np.ones(pts), sUnit) * absRad
        
    tSteps = np.linspace(0, 1, pts)
    sinSep = np.sin(angSep)
    
    gArc = np.zeros((pts, 3))
    for sIdx, tVal in enumerate(tSteps):
        sWeight = np.sin((1 - tVal) * angSep) / sinSep
        eWeight = np.sin(tVal * angSep) / sinSep
        gArc[sIdx] = (sWeight * sUnit + eWeight * eUnit) * absRad
    return gArc

def appendLoopClosure(tMat):
    return np.column_stack((tMat, tMat[:, 0]))


win = plt.figure(figsize=(10, 9))
ax = win.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

lim = pi / 4 
sRoll  = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'Roll (X)', -lim, lim, valinit=0)
sPitch = Slider(plt.axes([0.25, 0.10, 0.65, 0.03]), 'Pitch (Y)', -lim, lim, valinit=0)
sYaw   = Slider(plt.axes([0.25, 0.05, 0.65, 0.03]), 'Yaw (Z)', -lim, lim, valinit=0)

platTrackBase = np.copy(platHome)

def refreshVisuals(sVal):
    ax.cla() 
    
    qR = qExp(np.array([1, 0, 0]), sRoll.val)
    qP = qExp(np.array([0, 1, 0]), sPitch.val)
    qY = qExp(np.array([0, 0, 1]), sYaw.val)
    qComb = qMult(qY, qMult(qP, qR))
    
    pUpPos = np.zeros_like(platTrackBase)
    intJointPos = np.zeros_like(platTrackBase)
    
    for idx in range(3):
        pUpPos[:, idx] = qRot(qComb, platTrackBase[:, idx])
        uPlat = pUpPos[:, idx] / np.linalg.norm(pUpPos[:, idx])
        uBase = bUnitVecs[:, idx]
        
        pDot = np.dot(uBase, uPlat)            
        iLineDir = np.cross(uBase, uPlat)      
        
        pScalDenom = 1.0 - pDot * pDot
        if abs(pScalDenom) < 1e-6: 
            pScalDenom = 1e-6 if pScalDenom >= 0 else -1e-6
            
        midIntVec = ((cosAlpha - cosBeta * pDot) / pScalDenom) * uBase + ((cosBeta - cosAlpha * pDot) / pScalDenom) * uPlat
        
        lDirNorm = np.linalg.norm(iLineDir)
        uLineDir = iLineDir / lDirNorm if lDirNorm > 1e-6 else np.array([0.0, 0.0, 1.0])
        
        dispMag = np.sqrt(np.maximum(0.0, 1.0 - np.dot(midIntVec, midIntVec)))
        
        intJointPos[:, idx] = (midIntVec + dispMag * uLineDir) * outRad
   
    cMotAngs = np.zeros(3)
    for idx in range(3):
        pShift = idx * (120.0 * pi / 180.0)
        jCoords = intJointPos[:, idx]
        
        lX = jCoords[0] * np.cos(pShift) + jCoords[1] * np.sin(pShift)
        lY = -jCoords[0] * np.sin(pShift) + jCoords[1] * np.cos(pShift)
        
        cMotAngs[idx] = atan2Cordic(lY, lX) * 180.0 / pi
    
    for idx in range(3):
        pArcSeg = computeSlerpArc(baseHome[:, idx], intJointPos[:, idx], outRad)
        dArcSeg = computeSlerpArc(intJointPos[:, idx], pUpPos[:, idx], outRad)
        ax.plot(pArcSeg[:, 0], pArcSeg[:, 1], pArcSeg[:, 2], color='firebrick', linewidth=4)
        ax.plot(dArcSeg[:, 0], dArcSeg[:, 1], dArcSeg[:, 2], color='dodgerblue', linewidth=4)

    ax.scatter(baseHome[0, :], baseHome[1, :], baseHome[2, :], color='red', s=60, edgecolors='black')
    ax.scatter(pUpPos[0, :], pUpPos[1, :], pUpPos[2, :], color='blue', s=60, edgecolors='black')
    ax.scatter(intJointPos[0, :], intJointPos[1, :], intJointPos[2, :], color='green', s=60, edgecolors='black')

    clBase, clPlat = appendLoopClosure(baseHome), appendLoopClosure(pUpPos)
    ax.plot(clBase[0, :], clBase[1, :], clBase[2, :], color='red', linestyle='--', alpha=0.5)
    ax.plot(clPlat[0, :], clPlat[1, :], clPlat[2, :], color='blue', linestyle='--', alpha=0.5)

    ax.set_title(f"Generalized Agile Eye IK ({int(alphaDeg)}° / {int(betaDeg)}° Links)", fontsize=13)
    ax.set_xlim([-outRad, outRad])
    ax.set_ylim([-outRad, outRad])
    ax.set_zlim([-outRad, outRad])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    win.canvas.draw_idle()
    print(cMotAngs)

sRoll.on_changed(refreshVisuals)
sPitch.on_changed(refreshVisuals)
sYaw.on_changed(refreshVisuals)

refreshVisuals(0)
plt.show()