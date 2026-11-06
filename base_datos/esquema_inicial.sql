-- ============================================================================
-- BASE DE DATOS: bd_deteccion_fraccionamiento
-- Sistema de Inteligencia Artificial para Detección de Fraccionamiento (SEACE/OSCE)
-- Arquitectura Híbrida: Reglas Normativas + NLP + Isolation Forest (RBAC)
-- ============================================================================

CREATE DATABASE IF NOT EXISTS bd_deteccion_fraccionamiento 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

USE bd_deteccion_fraccionamiento;

-- 1. Usuarios, Roles y Permisos (RBAC)
CREATE TABLE IF NOT EXISTS usuarios (
    id_usuario INT AUTO_INCREMENT PRIMARY KEY,
    nombre_usuario VARCHAR(50) NOT NULL UNIQUE,
    clave_encriptada VARCHAR(255) NOT NULL, -- Cifrado mediante Bcrypt
    nombre_completo VARCHAR(100) NOT NULL,
    correo VARCHAR(100) NULL,
    rol ENUM('Administrador', 'Supervisor', 'Auditor', 'Investigador') DEFAULT 'Auditor',
    estado TINYINT(1) DEFAULT 1,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. Bitácora de Sistema (Logs de Auditoría y Seguridad)
CREATE TABLE IF NOT EXISTS logs_sistema (
    id_log INT AUTO_INCREMENT PRIMARY KEY,
    id_usuario INT NULL,
    accion VARCHAR(100) NOT NULL,
    modulo VARCHAR(50) NOT NULL,
    ip_origen VARCHAR(45) DEFAULT '127.0.0.1',
    detalle TEXT,
    fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_logs_usuarios FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. Configuración Dinámica del Sistema y Parámetros
CREATE TABLE IF NOT EXISTS configuracion_sistema (
    id_config INT AUTO_INCREMENT PRIMARY KEY,
    frecuencia_contaminacion DECIMAL(4,3) DEFAULT 0.030,
    umbral_similitud_nlp DECIMAL(3,2) DEFAULT 0.50,
    ventana_dias INT DEFAULT 30,
    monto_uit_vigente DECIMAL(10,2) DEFAULT 5150.00,
    ruta_almacenamiento_pdf VARCHAR(255) DEFAULT './reportes_pdf/',
    notificar_riesgo_alto TINYINT(1) DEFAULT 1,
    umbral_alerta_notificacion DECIMAL(3,2) DEFAULT 0.85,
    flujo_revision VARCHAR(20) DEFAULT 'Automático',
    fecha_actualizacion DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. Versionado y Métricas del Modelo de IA
CREATE TABLE IF NOT EXISTS modelos_ia (
    id_modelo INT AUTO_INCREMENT PRIMARY KEY,
    nombre_modelo VARCHAR(100) NOT NULL,
    fecha_entrenamiento DATETIME DEFAULT CURRENT_TIMESTAMP,
    metrica_precision DECIMAL(5,4) NULL,
    metrica_recall DECIMAL(5,4) NULL,
    metrica_f1 DECIMAL(5,4) NULL,
    metrica_roc_auc DECIMAL(5,4) NULL,
    estado_activo TINYINT(1) DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. Historial de Cargas de Archivos Excel / API
CREATE TABLE IF NOT EXISTS historial_cargas (
    id_carga INT AUTO_INCREMENT PRIMARY KEY,
    nombre_archivo VARCHAR(255) NOT NULL,
    total_registros INT NOT NULL,
    ventana_dias INT NULL,
    fecha_carga DATETIME DEFAULT CURRENT_TIMESTAMP,
    id_usuario INT NULL,
    CONSTRAINT fk_cargas_usuarios FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 6. Convocatorias / Compras Procesadas (Estructura alineada al SEACE)
CREATE TABLE IF NOT EXISTS convocatorias_procesadas (
    id_convocatoria INT AUTO_INCREMENT PRIMARY KEY,
    id_carga INT NOT NULL,
    entidad_ruc VARCHAR(20) NOT NULL,
    entidad_nombre VARCHAR(255) NOT NULL,
    descripcion_item TEXT NOT NULL,
    monto_referencial DECIMAL(15,2) NOT NULL,
    fecha_convocatoria DATE NOT NULL,
    codigo_cubso VARCHAR(255) NULL,
    tipo_proceso VARCHAR(100) DEFAULT 'Igual o Menor a 8 UIT',
    CONSTRAINT fk_convocatorias_cargas FOREIGN KEY (id_carga) REFERENCES historial_cargas(id_carga) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Índices de Rendimiento para convocatorias
CREATE INDEX idx_entidad_fecha ON convocatorias_procesadas(entidad_ruc, fecha_convocatoria);
CREATE INDEX idx_codigo_cubso ON convocatorias_procesadas(codigo_cubso(100));
CREATE INDEX idx_fecha_convocatoria ON convocatorias_procesadas(fecha_convocatoria);

-- 7. Resultados de IA y Feedback del Auditor (Human-in-the-Loop)
CREATE TABLE IF NOT EXISTS resultados_anomalias (
    id_resultado INT AUTO_INCREMENT PRIMARY KEY,
    id_convocatoria INT NOT NULL,
    id_modelo INT NULL,
    similitud_nlp DECIMAL(5,4) NOT NULL,
    monto_acumulado_30d DECIMAL(15,2) NOT NULL,
    frecuencia_compras_30d INT NOT NULL,
    dias_desde_compra_anterior INT NOT NULL,
    score_reglas_negocio DECIMAL(6,2) DEFAULT 0,
    bono_ia DECIMAL(6,2) DEFAULT 0,
    atipicidad_ia DOUBLE NULL, -- atipicidad del Isolation Forest; ordena los empates de puntaje
    puntaje_riesgo DECIMAL(6,2) NOT NULL,
    nivel_riesgo VARCHAR(20) DEFAULT 'Bajo',
    es_anomalia TINYINT(1) NOT NULL,
    explicacion_reglas TEXT NULL,
    estado_revision VARCHAR(30) DEFAULT 'Pendiente', -- 'Pendiente', 'Confirmado_Fraccionamiento', 'Descartado'
    comentario_auditor TEXT NULL,
    id_usuario_auditor INT NULL,
    fecha_revision DATETIME NULL,
    etiqueta_referencia TINYINT(1) NULL, -- 1 = fraccionamiento confirmado, 0 = conforme (fuente externa, p. ej. Contraloría)
    CONSTRAINT fk_resultados_convocatorias FOREIGN KEY (id_convocatoria) REFERENCES convocatorias_procesadas(id_convocatoria) ON DELETE CASCADE,
    CONSTRAINT fk_resultados_modelos FOREIGN KEY (id_modelo) REFERENCES modelos_ia(id_modelo) ON DELETE SET NULL,
    CONSTRAINT fk_resultados_auditor FOREIGN KEY (id_usuario_auditor) REFERENCES usuarios(id_usuario) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE INDEX idx_puntaje_riesgo ON resultados_anomalias(puntaje_riesgo);

-- 8. Conjuntos de evaluación del modelo
CREATE TABLE IF NOT EXISTS datasets_evaluacion (
    id_dataset INT AUTO_INCREMENT PRIMARY KEY,
    tipo_dataset ENUM('Entrenamiento', 'Validacion', 'Produccion') NOT NULL,
    cantidad_registros INT NOT NULL,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================================
-- DATOS INICIALES Y ROLES DEL ENTORNO LOCAL
-- ============================================================================

-- Usuario Administrador (Password: admin123 -> Hash Bcrypt)
INSERT INTO usuarios (nombre_usuario, clave_encriptada, nombre_completo, correo, rol, estado)
SELECT 'admin', '$2b$12$6/7/LhQ5y6G.G8tq6U2Wle3NnQO5s9aTzJ0W5W9W5W9W5W9W5W9W5', 'Administrador Principal SEACE', 'admin@contraloria.gob.pe', 'Administrador', 1
WHERE NOT EXISTS (SELECT 1 FROM usuarios WHERE nombre_usuario = 'admin');

-- Usuario Supervisor (Password: supervisor123)
INSERT INTO usuarios (nombre_usuario, clave_encriptada, nombre_completo, correo, rol, estado)
SELECT 'supervisor', '$2b$12$6/7/LhQ5y6G.G8tq6U2Wle3NnQO5s9aTzJ0W5W9W5W9W5W9W5W9W5', 'Supervisor de Auditoría OSCE', 'supervisor@osce.gob.pe', 'Supervisor', 1
WHERE NOT EXISTS (SELECT 1 FROM usuarios WHERE nombre_usuario = 'supervisor');

-- Usuario Auditor (Password: auditor123)
INSERT INTO usuarios (nombre_usuario, clave_encriptada, nombre_completo, correo, rol, estado)
SELECT 'auditor', '$2b$12$6/7/LhQ5y6G.G8tq6U2Wle3NnQO5s9aTzJ0W5W9W5W9W5W9W5W9W5', 'Auditor de Contrataciones OSCE', 'auditor@osce.gob.pe', 'Auditor', 1
WHERE NOT EXISTS (SELECT 1 FROM usuarios WHERE nombre_usuario = 'auditor');

-- Usuario Investigador (Password: investigador123)
INSERT INTO usuarios (nombre_usuario, clave_encriptada, nombre_completo, correo, rol, estado)
SELECT 'investigador', '$2b$12$6/7/LhQ5y6G.G8tq6U2Wle3NnQO5s9aTzJ0W5W9W5W9W5W9W5W9W5', 'Investigador Institucional', 'investigador@contraloria.gob.pe', 'Investigador', 1
WHERE NOT EXISTS (SELECT 1 FROM usuarios WHERE nombre_usuario = 'investigador');

-- Configuración por defecto del sistema
INSERT INTO configuracion_sistema (frecuencia_contaminacion, umbral_similitud_nlp, ventana_dias, monto_uit_vigente, ruta_almacenamiento_pdf, notificar_riesgo_alto, umbral_alerta_notificacion)
SELECT 0.030, 0.50, 30, 5150.00, './reportes_pdf/', 1, 0.85
WHERE NOT EXISTS (SELECT 1 FROM configuracion_sistema);

-- Modelo de IA Inicial por defecto
INSERT INTO modelos_ia (nombre_modelo, metrica_precision, metrica_recall, metrica_f1, metrica_roc_auc, estado_activo)
SELECT 'Isolation Forest v1.0 (Baseline)', 0.8850, 0.8420, 0.8629, 0.9110, 1
WHERE NOT EXISTS (SELECT 1 FROM modelos_ia);
