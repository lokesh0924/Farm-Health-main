import React from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  useTheme,
} from '@mui/material';
import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';

const About = () => {
  const { t } = useTranslation();
  const theme = useTheme();

  return (
    <Box
      sx={{
        background: 'linear-gradient(180deg, #f5f5f5 0%, #ffffff 100%)',
        minHeight: '100vh',
        py: 8,
      }}
    >
      <Container maxWidth="lg">
        <Box sx={{ mb: 8 }}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Typography
              variant="h2"
              component="h1"
              align="center"
              gutterBottom
              sx={{
                mb: 2,
                fontWeight: 700,
                background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                backgroundClip: 'text',
                textFillColor: 'transparent',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                textShadow: '0 2px 4px rgba(0,0,0,0.1)',
              }}
            >
              {t('title')}
            </Typography>
            <Typography
              variant="h6"
              align="center"
              color="text.secondary"
              sx={{
                mb: 6,
                maxWidth: '800px',
                mx: 'auto',
                lineHeight: 1.8,
                fontSize: '1.1rem',
                letterSpacing: '0.5px',
              }}
            >
              {t('description')}
            </Typography>
          </motion.div>

          <Grid container spacing={4} justifyContent="center">
            <Grid item xs={12} md={8}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <Card
                  elevation={3}
                  sx={{
                    height: '100%',
                    transition: 'all 0.3s ease-in-out',
                    '&:hover': {
                      transform: 'translateY(-5px)',
                      boxShadow: theme.shadows[8],
                    },
                    background: 'linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%)',
                    borderRadius: 2,
                  }}
                >
                  <CardContent sx={{ p: 4 }}>
                    <Typography
                      variant="h4"
                      gutterBottom
                      sx={{
                        fontWeight: 600,
                        color: theme.palette.primary.main,
                        mb: 3,
                        position: 'relative',
                        '&::after': {
                          content: '""',
                          position: 'absolute',
                          bottom: -8,
                          left: 0,
                          width: 60,
                          height: 3,
                          backgroundColor: theme.palette.primary.main,
                          borderRadius: 2,
                        },
                      }}
                    >
                      {t('mission')}
                    </Typography>
                    <Typography
                      variant="body1"
                      paragraph
                      sx={{ 
                        lineHeight: 1.8, 
                        mb: 2,
                        fontSize: '1.1rem',
                        color: 'text.primary',
                      }}
                    >
                      {t('mission_text')}
                    </Typography>
                    <Typography
                      variant="body1"
                      sx={{ 
                        lineHeight: 1.8,
                        fontSize: '1.1rem',
                        color: 'text.primary',
                      }}
                    >
                      {t('mission_text_2')}
                    </Typography>
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          </Grid>
        </Box>
      </Container>
    </Box>
  );
};

export default About;
